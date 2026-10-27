/* This file is part of the Spring engine (GPL v2 or later), see LICENSE.html */

#include "Trackpad.h"
#include "System/Config/ConfigHandler.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <type_traits>

#include <CoreGraphics/CGGeometry.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <SDL_syswm.h>
#include <SDL_timer.h>
#include <SDL_video.h>

namespace MacTrackpad {
namespace {

constexpr unsigned long SCROLL_WHEEL = 22;
constexpr unsigned long MAGNIFY = 30;
constexpr unsigned long BEGAN = 1;
constexpr unsigned long ENDED = 8;
constexpr unsigned long CANCELLED = 16;
constexpr unsigned long TOUCH_PRESENT = 7;
constexpr unsigned long OPTION = 1 << 19;

bool enabled = false;
SDL_Window* sdlWindow = nullptr;
id nativeWindow = nullptr;
id contentView = nullptr;
Uint32 gestureType = Uint32(-1);
std::array<id, 3> previousTouches = {};
CGPoint previousCenter = {};
bool tilting = false;
bool scrollTilts = false;
bool scrolling = false;
bool pinching = false;
bool suppressScroll = false;

using TouchCallback = void (*)(id, SEL, id);
struct TouchHook {
	const char* selector;
	TouchCallback original = nullptr;
};
std::array<TouchHook, 4> touchHooks = {{
	{"touchesBeganWithEvent:"}, {"touchesMovedWithEvent:"},
	{"touchesEndedWithEvent:"}, {"touchesCancelledWithEvent:"},
}};
Class hookedViewClass = nullptr;

template<typename R = id, typename... Args>
R Send(id object, const char* selector, Args... args)
{
	return reinterpret_cast<R (*)(id, SEL, Args...)>(objc_msgSend)(object, sel_registerName(selector), args...);
}

bool Focused()
{
	return sdlWindow != nullptr && (SDL_GetWindowFlags(sdlWindow) & SDL_WINDOW_INPUT_FOCUS) != 0;
}

void Push(Action action, float x, float y, bool begin = false, bool end = false, bool momentum = false)
{
	static_assert(std::is_trivially_copyable_v<Gesture> && sizeof(Gesture) <= sizeof(SDL_Event));
	if (gestureType == Uint32(-1) || sdlWindow == nullptr)
		return;

	const Gesture gesture = {gestureType, SDL_GetTicks(), SDL_GetWindowID(sdlWindow), action, x, y, begin, end, momentum};
	SDL_Event event = {};
	std::memcpy(&event, &gesture, sizeof(gesture));
	SDL_PushEvent(&event);
}

void FinishTilt()
{
	if (tilting)
		Push(Action::Tilt, 0, 0, false, true);
	tilting = false;
	previousTouches = {};
}

void Touches(id self, SEL selector, id event)
{
	if (enabled && self == contentView && Focused()) {
		const bool cancelled = selector == sel_registerName("touchesCancelledWithEvent:");
		id touches = Send(event, "touchesMatchingPhase:inView:", TOUCH_PRESENT, self);
		id objects = Send(touches, "allObjects");
		const auto count = Send<unsigned long>(objects, "count");
		std::array<id, 3> identities = {};
		CGPoint center = {};
		unsigned int fingers = 0;

		for (unsigned long i = 0; i < count; ++i) {
			id touch = Send(objects, "objectAtIndex:", i);
			// Ignore resting palms and direct touch screens.
			if (Send<signed char>(touch, "isResting") || Send<long>(touch, "type") != 1)
				continue;
			if (fingers < identities.size())
				identities[fingers] = Send(touch, "identity");
			const CGPoint point = Send<CGPoint>(touch, "normalizedPosition");
			center.x += point.x;
			center.y += point.y;
			++fingers;
		}

		if (fingers == 3 && !cancelled && configHandler->GetBool("MacTrackpadThreeFingerTilt")) {
			std::sort(identities.begin(), identities.end(), std::less<id>());
			center.x /= 3;
			center.y /= 3;
			if (tilting && identities == previousTouches) {
				// NSTouch positions increase upward, opposite to trackpad scroll deltas.
				Push(Action::Tilt, 0, float(previousCenter.y - center.y) * 800.0f);
			} else {
				FinishTilt();
				Push(Action::Tilt, 0, 0, true);
			}
			tilting = true;
			suppressScroll = true;
			previousTouches = identities;
			previousCenter = center;
		} else {
			FinishTilt();
			// Do not turn the remaining two fingers into a camera pan.
			if (fingers == 0 || cancelled)
				suppressScroll = false;
		}
	}

	// Preserve SDL's own mouse/touch handling, including taps and click-drag.
	for (const TouchHook& hook: touchHooks) {
		if (selector == sel_registerName(hook.selector) && hook.original != nullptr) {
			hook.original(self, selector, event);
			break;
		}
	}
}

}

void Init(SDL_Window* window)
{
	sdlWindow = window;
	SDL_SysWMinfo info = {};
	SDL_VERSION(&info.version);
	if (!SDL_GetWindowWMInfo(window, &info) || info.subsystem != SDL_SYSWM_COCOA)
		return;

	nativeWindow = reinterpret_cast<id>(info.info.cocoa.window);
	contentView = Send(nativeWindow, "contentView");
	if (gestureType == Uint32(-1))
		gestureType = SDL_RegisterEvents(1);

	Class viewClass = object_getClass(contentView);
	if (hookedViewClass == nullptr) {
		for (TouchHook& hook: touchHooks) {
			const SEL selector = sel_registerName(hook.selector);
			Method method = class_getInstanceMethod(viewClass, selector);
			if (method == nullptr)
				continue;
			hook.original = reinterpret_cast<TouchCallback>(method_getImplementation(method));
			class_replaceMethod(viewClass, selector, reinterpret_cast<IMP>(&Touches), method_getTypeEncoding(method));
		}
		hookedViewClass = viewClass;
	}

	Send<void>(contentView, "setAcceptsTouchEvents:", static_cast<signed char>(1));
	Send<void>(contentView, "setAllowedTouchTypes:", 1UL << 1); // NSTouchTypeIndirect
	Reset();
}

void Reset()
{
	previousTouches = {};
	tilting = false;
	scrolling = false;
	pinching = false;
	scrollTilts = false;
	suppressScroll = false;
}

void SetEnabled(bool value)
{
	if (value != enabled)
		Reset();
	enabled = value;
}

bool HandleNativeEvent(id event)
{
	if (!enabled || !Focused() || Send(event, "window") != nativeWindow)
		return false;

	const auto type = Send<unsigned long>(event, "type");
	if (type == MAGNIFY) {
		const auto phase = Send<unsigned long>(event, "phase");
		const bool end = (phase & (ENDED | CANCELLED)) != 0;
		Push(Action::Pinch, 0, end ? 0.0f : float(Send<double>(event, "magnification")), !pinching || (phase & BEGAN) != 0, end);
		pinching = !end;
		return true;
	}

	if (type != SCROLL_WHEEL || !Send<signed char>(event, "hasPreciseScrollingDeltas"))
		return false; // Physical mouse wheels keep their usual zoom behavior.

	const auto phase = Send<unsigned long>(event, "phase");
	const auto momentum = Send<unsigned long>(event, "momentumPhase");
	const bool begin = !scrolling || (phase & BEGAN) != 0;
	const bool end = ((phase | momentum) & (ENDED | CANCELLED)) != 0;
	if (begin) {
		scrollTilts = (Send<unsigned long>(event, "modifierFlags") & OPTION) != 0;
	}

	if (!suppressScroll && !tilting && !pinching) {
		Push(scrollTilts ? Action::Tilt : Action::Pan,
			float(Send<double>(event, "scrollingDeltaX")),
			float(Send<double>(event, "scrollingDeltaY")), begin, end, momentum != 0);
	}
	// Keep the route until momentum ends (its first event follows phase=Ended).
	scrolling = !(momentum != 0 && end);
	return true; // Avoid delivering a second, rounded SDL wheel event.
}

bool GetGesture(const SDL_Event& event, Gesture& gesture)
{
	if (event.type != gestureType || gestureType == Uint32(-1))
		return false;
	std::memcpy(&gesture, &event, sizeof(gesture));
	return true;
}

}
