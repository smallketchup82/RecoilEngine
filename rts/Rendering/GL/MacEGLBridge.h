/* This file is part of the Recoil engine (GPL v2 or later), see LICENSE.html */

#pragma once

// Renders through Mesa EGL + Zink and presents through reusable Metal textures.
// Patched Mesa also supports shared GPU textures; SDL remains the upload fallback.

#if defined(__APPLE__) && defined(MAC_EGL_ZINK) && !defined(HEADLESS)
	#define MAC_EGL_BRIDGE_AVAILABLE 1
#endif

#ifdef MAC_EGL_BRIDGE_AVAILABLE
struct SDL_Window;

namespace MacEGLBridge {
	// true after a context was successfully created
	bool Active();

	void* CreateContext(SDL_Window* window, int major, int minor, bool core);
	void MakeCurrent(bool clear);
	void DestroyContext();
	void* GetProcAddress(const char* name);

	// presents the pbuffer and resizes it to track the window
	void SwapWindow(SDL_Window* window);
	void SetSwapInterval(int interval);
	void WaitForFrame();

	// The world uses logical resolution; the final UI uses backing pixels.
	void BeginScene();
	void BeginUI();
	int GetFontRasterScale();
	int GetRetinaMode();
	void ScaleSceneSize(int& width, int& height);
	void CopyToTexturePixels(unsigned int target, int level, int xoff, int yoff, int x, int y, int width, int height);
}

// Objective-C++ boundary: the engine uses GCC, MetalFX uses Apple's compiler.
namespace MacMetalFX {
	bool WaitForFrame(void* layer, int refreshRate, bool visible = true);
	void StopDisplayLink();

	bool Upscale(const void* rgba, int width, int height, int outputWidth, int outputHeight, const void*& output, int& outputRowLength);
	// A native Mesa queue enables GPU event synchronization. Without it, wait on the CPU.
	bool UpscaleShared(void* device, void* input, void* output, void* nativeQueue = nullptr);
	bool PresentShared(void* device, void* input, void* layer, float headroom = 0.0f, void* nativeQueue = nullptr);
	bool PresentPixels(const void* rgba, int width, int height, void* layer);
	// Drain before releasing borrowed Mesa allocations, resizing, or changing paths.
	void WaitShared();
	bool ConfigureHDR(void* layer, void* nativeWindow, bool enabled);
	float HDRHeadroom(void* nativeWindow);
	const char* GetError();
	void Destroy();
}
#endif
