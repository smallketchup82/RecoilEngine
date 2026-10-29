/* This file is part of the Spring engine (GPL v2 or later), see LICENSE.html */

#include "System/Platform/WindowManagerHelper.h"
#include <SDL_video.h>
#include <SDL_syswm.h>
#include <CoreGraphics/CGGeometry.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>


namespace WindowManagerHelper {

void BlockCompositing(SDL_Window* window)
{
	//FIXME implement?
}


int GetWindowState(SDL_Window* window)
{
	return SDL_GetWindowFlags(window);
}


template<typename R = id, typename... Args>
static R Send(id object, const char* selector, Args... args)
{
	return reinterpret_cast<R (*)(id, SEL, Args...)>(objc_msgSend)(object, sel_registerName(selector), args...);
}

// The launcher owns Sparkle; route this engine menu action to that launch only.
static void PostUpdateRequest(const char* notificationName)
{
	const char* requestID = std::getenv("ZEROK_UPDATE_REQUEST_ID");
	if (requestID == nullptr)
		return;
	id name = Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", notificationName);
	id object = Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", requestID);
	id center = Send(reinterpret_cast<id>(objc_getClass("NSDistributedNotificationCenter")), "defaultCenter");
	Send<void>(center, "postNotificationName:object:userInfo:deliverImmediately:", name, object, static_cast<id>(nullptr), true);
}

static id automaticChecksItem = nullptr;

static void CheckForUpdates(id, SEL, id)
{
	PostUpdateRequest("org.zerok.macos.checkForUpdates");
}

static void ToggleAutomaticChecks(id, SEL, id)
{
	PostUpdateRequest("org.zerok.macos.toggleAutomaticChecks");
}

static void AutomaticChecksChanged(id, SEL, id notification)
{
	id enabled = Send(Send(notification, "userInfo"), "objectForKey:",
		Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", "enabled"));
	Send<void>(automaticChecksItem, "setState:", Send<bool>(enabled, "boolValue") ? static_cast<long>(1) : static_cast<long>(0));
}

void AddUpdateMenuItem()
{
	if (std::getenv("ZEROK_UPDATE_REQUEST_ID") == nullptr)
		return;
	id app = Send(reinterpret_cast<id>(objc_getClass("NSApplication")), "sharedApplication");
	id menu = Send(Send(Send(app, "mainMenu"), "itemAtIndex:", static_cast<long>(0)), "submenu");
	if (menu == nullptr)
		return;
	static id target = [] {
		Class type = objc_allocateClassPair(objc_getClass("NSObject"), "ZeroKUpdateMenuTarget", 0);
		class_addMethod(type, sel_registerName("checkForUpdates:"), reinterpret_cast<IMP>(CheckForUpdates), "v@:@");
		class_addMethod(type, sel_registerName("toggleAutomaticChecks:"), reinterpret_cast<IMP>(ToggleAutomaticChecks), "v@:@");
		class_addMethod(type, sel_registerName("automaticChecksChanged:"), reinterpret_cast<IMP>(AutomaticChecksChanged), "v@:@");
		objc_registerClassPair(type);
		return Send(Send(reinterpret_cast<id>(type), "alloc"), "init");
	}();
	// Window recreation must not add duplicate menu items.
	if (Send<long>(menu, "indexOfItemWithTarget:andAction:", target, sel_registerName("checkForUpdates:")) >= 0)
		return;
	id title = Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", "Check for Updates…");
	id empty = Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", "");
	id item = Send(Send(reinterpret_cast<id>(objc_getClass("NSMenuItem")), "alloc"), "initWithTitle:action:keyEquivalent:", title, sel_registerName("checkForUpdates:"), empty);
	Send<void>(item, "setTarget:", target);
	Send<void>(menu, "insertItem:atIndex:", item, static_cast<long>(1));
	Send<void>(item, "release");
	id automaticTitle = Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", "Check Automatically");
	automaticChecksItem = Send(Send(reinterpret_cast<id>(objc_getClass("NSMenuItem")), "alloc"), "initWithTitle:action:keyEquivalent:", automaticTitle, sel_registerName("toggleAutomaticChecks:"), empty);
	Send<void>(automaticChecksItem, "setTarget:", target);
	const char* enabled = std::getenv("ZEROK_AUTOMATIC_UPDATE_CHECKS");
	Send<void>(automaticChecksItem, "setState:", static_cast<long>(enabled != nullptr && enabled[0] == '1'));
	Send<void>(menu, "insertItem:atIndex:", automaticChecksItem, static_cast<long>(2));
	id center = Send(reinterpret_cast<id>(objc_getClass("NSDistributedNotificationCenter")), "defaultCenter");
	id name = Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", "org.zerok.macos.automaticChecksChanged");
	id object = Send(reinterpret_cast<id>(objc_getClass("NSString")), "stringWithUTF8String:", std::getenv("ZEROK_UPDATE_REQUEST_ID"));
	Send<void>(center, "addObserver:selector:name:object:", target, sel_registerName("automaticChecksChanged:"), name, object);
	PostUpdateRequest("org.zerok.macos.requestAutomaticChecks");
}

bool GetMacNotchGeometry(SDL_Window* window, double& left, double& right, double& height)
{
	if (window == nullptr || (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) == 0)
		return false;

	SDL_SysWMinfo info = {};
	SDL_VERSION(&info.version);
	if (!SDL_GetWindowWMInfo(window, &info) || info.subsystem != SDL_SYSWM_COCOA)
		return false;

	id nativeWindow = reinterpret_cast<id>(info.info.cocoa.window);
	id screen = Send(nativeWindow, "screen");
	if (screen == nullptr)
		return false;

	const CGRect frame = Send<CGRect>(screen, "frame");
	const CGRect nativeFrame = Send<CGRect>(nativeWindow, "frame");
	const CGRect leftArea = Send<CGRect>(screen, "auxiliaryTopLeftArea");
	const CGRect rightArea = Send<CGRect>(screen, "auxiliaryTopRightArea");
	int width, windowHeight;
	SDL_GetWindowSize(window, &width, &windowHeight);
	if (leftArea.size.width <= 0 || rightArea.size.width <= 0 ||
		std::abs(frame.origin.x - nativeFrame.origin.x) > 1 ||
		std::abs(frame.origin.y - nativeFrame.origin.y) > 1 ||
		std::abs(width - frame.size.width) > 1 || std::abs(windowHeight - frame.size.height) > 1)
		return false;

	left = leftArea.origin.x + leftArea.size.width - frame.origin.x;
	right = rightArea.origin.x - frame.origin.x;
	height = std::min(leftArea.size.height, rightArea.size.height);
	return left > 0 && right > left && right < width && height > 0 && height < windowHeight;
}

void SetWindowResizable(SDL_Window* window, bool resizable)
{
	//FIXME implement?
}

}; // namespace WindowManagerHelper
