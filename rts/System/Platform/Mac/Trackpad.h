/* This file is part of the Spring engine (GPL v2 or later), see LICENSE.html */

#pragma once

#include <SDL_events.h>
#include <objc/objc.h>

struct SDL_Window;

namespace MacTrackpad {

enum class Action { Pan, Pinch, Tilt };

// A private SDL event carries these values by copy, without owning pointers.
struct Gesture {
	Uint32 type;
	Uint32 timestamp;
	Uint32 windowID;
	Action action;
	float x;
	float y;
	bool begin;
	bool end;
	bool momentum;
};

void Init(SDL_Window* window);
void SetEnabled(bool enabled);
void Reset();
bool HandleNativeEvent(id event);
bool GetGesture(const SDL_Event& event, Gesture& gesture);

}
