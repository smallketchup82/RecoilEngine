/* This file is part of the Recoil engine (GPL v2 or later), see LICENSE.html */

#include "MacEGLBridge.h"

#ifdef MAC_EGL_BRIDGE_AVAILABLE

#include <dlfcn.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <mutex>
#include <cmath>
#include <vector>

#include <SDL.h>
#include <SDL_syswm.h>

#include "Rendering/GL/myGL.h"
#include "Rendering/GlobalRendering.h"
#include "System/FileSystem/FileSystem.h"
#include "System/Log/ILog.h"
#include "System/Platform/Misc.h"
#include "System/Config/ConfigHandler.h"

CONFIG(int, MacRetinaMode).defaultValue(1).minimumValue(0).maximumValue(2).description("macOS Retina mode: 0=standard, 1=Retina text and UI, 2=Retina text, UI and main game scene. Requires restart.");
CONFIG(bool, MacMetalFXSpatial).defaultValue(false).description("Use MetalFX spatial upscaling for the lower-resolution world in Retina text and UI mode. Applies immediately; UI stays native resolution. Uses shared GPU textures when available, with a CPU staging compatibility path.");
CONFIG(bool, MacHDR).defaultValue(false).description("macOS HDR display output for the game's existing HDR lighting. Preserves floating-point scene values without extra bloom or color grading. Applies immediately; UI white remains SDR. Requires GPU sharing and an HDR display.");
CONFIG(bool, MacGPUSharing).defaultValue(false).description("Experimental sharing of bundled Mesa GPU textures with MetalFX and Metal presentation. Applies immediately. Disabled by default while synchronization performance is evaluated.");

CONFIG(bool, MacDisplayLink).defaultValue(false).description("Experimental macOS frame pacing with CAMetalDisplayLink before polling input. Disabled by default pending consistent performance improvements.");

// minimal EGL declarations; libEGL is dlopen()ed so there is no build-time dependency on Mesa
namespace {
	typedef void* EGLDisplay;
	typedef void* EGLConfig;
	typedef void* EGLSurface;
	typedef void* EGLContext;
	typedef int32_t EGLint;
	typedef unsigned int EGLBoolean;
	typedef unsigned int EGLenum;

	constexpr EGLint EGL_NONE = 0x3038;
	constexpr EGLint EGL_RED_SIZE = 0x3024, EGL_GREEN_SIZE = 0x3023, EGL_BLUE_SIZE = 0x3022, EGL_ALPHA_SIZE = 0x3021;
	constexpr EGLint EGL_DEPTH_SIZE = 0x3025, EGL_STENCIL_SIZE = 0x3026;
	constexpr EGLint EGL_SURFACE_TYPE = 0x3033, EGL_PBUFFER_BIT = 0x0001;
	constexpr EGLint EGL_RENDERABLE_TYPE = 0x3040, EGL_OPENGL_BIT = 0x0008;
	constexpr EGLint EGL_WIDTH = 0x3057, EGL_HEIGHT = 0x3056;
	constexpr EGLenum EGL_OPENGL_API = 0x30A2;
	constexpr EGLint EGL_CONTEXT_MAJOR_VERSION = 0x3098, EGL_CONTEXT_MINOR_VERSION = 0x30FB;
	constexpr EGLint EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD;
	constexpr EGLint EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x1, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT = 0x2;
	constexpr EGLenum EGL_PLATFORM_SURFACELESS_MESA = 0x31DD;

	struct EGLFuncs {
		void* lib = nullptr;
		void* (*GetProcAddress)(const char*) = nullptr;
		EGLDisplay (*GetPlatformDisplayEXT)(EGLenum, void*, const EGLint*) = nullptr;
		EGLBoolean (*Initialize)(EGLDisplay, EGLint*, EGLint*) = nullptr;
		EGLBoolean (*Terminate)(EGLDisplay) = nullptr;
		EGLBoolean (*BindAPI)(EGLenum) = nullptr;
		EGLBoolean (*ChooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*) = nullptr;
		EGLSurface (*CreatePbufferSurface)(EGLDisplay, EGLConfig, const EGLint*) = nullptr;
		EGLBoolean (*DestroySurface)(EGLDisplay, EGLSurface) = nullptr;
		EGLContext (*CreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint*) = nullptr;
		EGLBoolean (*DestroyContext)(EGLDisplay, EGLContext) = nullptr;
		EGLBoolean (*MakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext) = nullptr;
		EGLint (*GetError)() = nullptr;
	} egl;

	EGLDisplay display = nullptr;
	EGLConfig config = nullptr;
	EGLContext context = nullptr;
	EGLSurface surface = nullptr;
	int surfSizeX = 0;
	int surfSizeY = 0;

	SDL_Renderer* renderer = nullptr;
	SDL_Texture* texture = nullptr;
	int texSizeX = 0;
	int texSizeY = 0;
	SDL_Window* presenterWindow = nullptr;
	bool retinaEnabled = false;
	int retinaMode = 0;
	bool sharedUpscaleFailed = false;
	bool sharedPresentFailed = false;
	bool sharedPresentLogged = false;
	bool presentedSharedLastFrame = false;
	bool displayLinkLogged = false;
	bool pixelPresentFailed = false;
	bool pixelPresentLogged = false;

	bool hdrFrame = false, hdrFailed = false, hdrLogged = false;
	bool hdrSceneDrawn = false, previousHDRRequest = false, hdrUnavailableLogged = false;
	float loggedHDRHeadroom = 0.0f;

	void* NativeWindow() {
		SDL_SysWMinfo info = {};
		SDL_VERSION(&info.version);
		if (!SDL_GetWindowWMInfo(presenterWindow, &info) || info.subsystem != SDL_SYSWM_COCOA)
			return nullptr;
		return info.info.cocoa.window;
	}

	using GetVkColorBuffer = bool (*)(void**, uint64_t*);
	using GetMetalTexture = bool (*)(void*, uint64_t, void**, void**);
	GetVkColorBuffer getVkColorBuffer = nullptr;
	GetMetalTexture getMetalTexture = nullptr;
	using GetMetalQueue = void* (*)(void*);
	GetMetalQueue getMetalQueue = nullptr;
	bool (*flushGPU)() = nullptr;
	bool sharedEventsLogged = false;
	bool sharedGPUWasEnabled = false;
	void* sharedDriver = nullptr;
	bool sharingChecked = false;

	constexpr unsigned int NUM_PBOS = 2;
	GLuint pbos[NUM_PBOS] = {};
	GLsync fences[NUM_PBOS] = {};
	int pboSizes[NUM_PBOS][2] = {};
	unsigned int frameIdx = 0;

	std::mutex surfMutex;

	// a Mesa tree shipped next to the engine binary takes precedence over the system one
	const std::string& MesaLibraryDirectory() {
		static const std::string prefix = []() -> std::string {
			const std::string bundled = Platform::GetProcessExecutablePath() + "../Frameworks";
			if (FileSystem::FileExists(bundled + "/libEGL.1.dylib")) return bundled;
			const char* developmentPrefix = std::getenv("RECOIL_MESA_PREFIX");
			return developmentPrefix != nullptr ? std::string(developmentPrefix) + "/lib" : Platform::GetProcessExecutablePath() + "mesa/lib";
		}();
		return prefix;
	}

	bool LoadEGL() {
		if (egl.lib != nullptr)
			return true;

		if ((egl.lib = dlopen((MesaLibraryDirectory() + "/libEGL.1.dylib").c_str(), RTLD_NOW | RTLD_LOCAL)) == nullptr) {
			LOG_L(L_ERROR, "[MacEGLBridge] could not load Mesa libEGL (%s)", dlerror());
			return false;
		}

		#define LOADSYM(n) egl.n = reinterpret_cast<decltype(egl.n)>(dlsym(egl.lib, "egl" #n))
		LOADSYM(GetProcAddress);
		LOADSYM(Initialize);
		LOADSYM(Terminate);
		LOADSYM(BindAPI);
		LOADSYM(ChooseConfig);
		LOADSYM(CreatePbufferSurface);
		LOADSYM(DestroySurface);
		LOADSYM(CreateContext);
		LOADSYM(DestroyContext);
		LOADSYM(MakeCurrent);
		LOADSYM(GetError);
		#undef LOADSYM
		if (egl.GetProcAddress != nullptr)
			egl.GetPlatformDisplayEXT = reinterpret_cast<decltype(egl.GetPlatformDisplayEXT)>(egl.GetProcAddress("eglGetPlatformDisplayEXT"));

		return (egl.GetPlatformDisplayEXT != nullptr && egl.Initialize != nullptr && egl.CreateContext != nullptr && egl.MakeCurrent != nullptr);
	}

	bool SharedExportsAvailable() {
		if (!sharingChecked) {
			sharingChecked = true;
			getVkColorBuffer = reinterpret_cast<GetVkColorBuffer>(dlsym(egl.lib, "recoil_mesa_get_vk_color_buffer_v1"));
			flushGPU = reinterpret_cast<decltype(flushGPU)>(dlsym(egl.lib, "recoil_mesa_flush_gpu_v1"));
			// Only use the ICD already loaded from this same Mesa bundle.
			sharedDriver = dlopen((MesaLibraryDirectory() + "/libvulkan_kosmickrisp.dylib").c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
			if (sharedDriver) {
				getMetalTexture = reinterpret_cast<GetMetalTexture>(dlsym(sharedDriver, "recoil_kk_get_metal_texture_v1"));
				getMetalQueue = reinterpret_cast<GetMetalQueue>(dlsym(sharedDriver, "recoil_kk_get_metal_queue_v1"));
			}
		}
		return getVkColorBuffer && getMetalTexture;
	}

	bool UseSharedGPU() {
		if (configHandler->GetBool("MacGPUSharing") || hdrFrame) {
			sharedGPUWasEnabled = true;
			return true;
		}
		if (sharedGPUWasEnabled)
			MacMetalFX::WaitShared();
		sharedGPUWasEnabled = false;
		sharedUpscaleFailed = sharedPresentFailed = false;
		sharedPresentLogged = false;
		return false;
	}

	bool GetSharedColorBuffer(void*& device, void*& nativeTexture, void** nativeQueue = nullptr) {
		void* vkDevice = nullptr;
		uint64_t image = 0;
		if (!getVkColorBuffer(&vkDevice, &image) || !getMetalTexture(vkDevice, image, &device, &nativeTexture))
			return false;
		if (nativeQueue)
			*nativeQueue = getMetalQueue ? getMetalQueue(vkDevice) : nullptr;
		return true;
	}

	void* SubmitSharedGPU(void* nativeQueue) {
		if (nativeQueue && flushGPU && flushGPU()) {
			if (!sharedEventsLogged)
				LOG("[MacGPUSharing] GPU event synchronization enabled; no per-frame CPU completion waits");
			sharedEventsLogged = true;
			return nativeQueue;
		}
		glFinish();
		return nullptr;
	}

	void SetupEnvironment() {
		setenv("GALLIUM_DRIVER", "zink", 0);

		// Zink needs robustness2.nullDescriptor, which MoltenVK lacks, so the loader gets KosmicKrisp
		if (getenv("VK_ICD_FILENAMES") == nullptr && getenv("VK_DRIVER_FILES") == nullptr) {
			const std::string bundled = Platform::GetProcessExecutablePath() + "../Resources/kosmickrisp_mesa_icd.aarch64.json";
			const std::string icd = FileSystem::FileExists(bundled) ? bundled : MesaLibraryDirectory() + "/../share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json";
			setenv("VK_DRIVER_FILES", icd.c_str(), 0);
		}

		// Zink reports GL 2.1 on KosmicKrisp due to missing geometry shaders
		setenv("MESA_GL_VERSION_OVERRIDE", "4.6COMPAT", 0);
		setenv("MESA_GLSL_VERSION_OVERRIDE", "460", 0);
	}

	bool CreateSurface(int sx, int sy) {
		MacMetalFX::WaitShared();
		pixelPresentFailed = false;
		sx = std::max(sx, 1);
		sy = std::max(sy, 1);

		const EGLint attrs[] = {EGL_WIDTH, sx, EGL_HEIGHT, sy, EGL_NONE};
		EGLSurface s = egl.CreatePbufferSurface(display, config, attrs);

		if (s == nullptr) {
			LOG_L(L_ERROR, "[MacEGLBridge] eglCreatePbufferSurface(%dx%d) failed: 0x%x", sx, sy, egl.GetError());
			return false;
		}

		EGLSurface old = surface;
		surface = s;
		surfSizeX = sx;
		surfSizeY = sy;

		if (context != nullptr)
			egl.MakeCurrent(display, surface, surface, context);
		if (old != nullptr)
			egl.DestroySurface(display, old);

		return true;
	}

	bool CreatePresenter(SDL_Window* window) {
		if (renderer != nullptr)
			return true;

		SDL_SetHint(SDL_HINT_RENDER_DRIVER, "metal");
		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
		if (renderer == nullptr) {
			LOG_L(L_ERROR, "[MacEGLBridge] SDL_CreateRenderer failed: %s", SDL_GetError());
			return false;
		}

		SDL_RendererInfo info;
		if (SDL_GetRendererInfo(renderer, &info) == 0)
			LOG("[MacEGLBridge] presenting through SDL renderer \"%s\"", info.name);

		int wx = 0, wy = 0, px = 0, py = 0;
		SDL_GetWindowSize(window, &wx, &wy);
		SDL_GetRendererOutputSize(renderer, &px, &py);
		LOG("[MacEGLBridge] logical=%dx%d output=%dx%d pbuffer=%dx%d", wx, wy, px, py, surfSizeX, surfSizeY);

		return true;
	}
}


// KosmicKrisp has neither wideLines nor geometry shaders, so Zink clamps every line to one pixel.
// Wide lines are approximated by repeating the draw with the viewport shifted by whole pixels.
namespace WideLines {
	thread_local float lineWidth = 1.0f;

	PFNGLLINEWIDTHPROC realLineWidth = nullptr;
	PFNGLDRAWARRAYSPROC realDrawArrays = nullptr;
	PFNGLDRAWELEMENTSPROC realDrawElements = nullptr;
	PFNGLDRAWRANGEELEMENTSPROC realDrawRangeElements = nullptr;
	PFNGLDRAWARRAYSINSTANCEDPROC realDrawArraysInstanced = nullptr;
	PFNGLDRAWELEMENTSINSTANCEDPROC realDrawElementsInstanced = nullptr;
	PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC realDrawArraysInstancedBaseInstance = nullptr;
	PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC realDrawElementsInstancedBaseInstance = nullptr;
	PFNGLDRAWELEMENTSBASEVERTEXPROC realDrawElementsBaseVertex = nullptr;
	PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC realDrawElementsInstancedBaseVertex = nullptr;
	PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC realDrawElementsInstancedBaseVertexBaseInstance = nullptr;

	bool IsLineMode(GLenum mode) {
		switch (mode) {
			case GL_LINES:
			case GL_LINE_STRIP:
			case GL_LINE_LOOP:
			case GL_LINES_ADJACENCY:
			case GL_LINE_STRIP_ADJACENCY:
				return true;
			default:
				return false;
		}
	}

	template<typename F>
	void Draw(GLenum mode, F&& draw) {
		draw();

		if (lineWidth < 1.5f || !IsLineMode(mode))
			return;

		// glPopAttrib restores width and viewport behind our back, so query both before redrawing
		GLfloat width = 1.0f;
		glGetFloatv(GL_LINE_WIDTH, &width);
		if (width < 1.5f)
			return;

		GLint viewport[4] = {};
		glGetIntegerv(GL_VIEWPORT, viewport);

		const int radius = static_cast<int>((width - 1.0f) * 0.5f + 0.5f);
		for (int i = 1; i <= radius; ++i) {
			const int offsets[4][2] = {{i, 0}, {-i, 0}, {0, i}, {0, -i}};
			for (const auto& o: offsets) {
				glViewport(viewport[0] + o[0], viewport[1] + o[1], viewport[2], viewport[3]);
				draw();
			}
		}
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
	}

	void APIENTRY LineWidth(GLfloat width) {
		lineWidth = width;
		realLineWidth(width);
	}
	void APIENTRY DrawArrays(GLenum mode, GLint first, GLsizei count) {
		Draw(mode, [&]() { realDrawArrays(mode, first, count); });
	}
	void APIENTRY DrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
		Draw(mode, [&]() { realDrawElements(mode, count, type, indices); });
	}
	void APIENTRY DrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices) {
		Draw(mode, [&]() { realDrawRangeElements(mode, start, end, count, type, indices); });
	}
	void APIENTRY DrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instances) {
		Draw(mode, [&]() { realDrawArraysInstanced(mode, first, count, instances); });
	}
	void APIENTRY DrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances) {
		Draw(mode, [&]() { realDrawElementsInstanced(mode, count, type, indices, instances); });
	}
	void APIENTRY DrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instances, GLuint baseInstance) {
		Draw(mode, [&]() { realDrawArraysInstancedBaseInstance(mode, first, count, instances, baseInstance); });
	}
	void APIENTRY DrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances, GLuint baseInstance) {
		Draw(mode, [&]() { realDrawElementsInstancedBaseInstance(mode, count, type, indices, instances, baseInstance); });
	}
	void APIENTRY DrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLint baseVertex) {
		Draw(mode, [&]() { realDrawElementsBaseVertex(mode, count, type, indices, baseVertex); });
	}
	void APIENTRY DrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances, GLint baseVertex) {
		Draw(mode, [&]() { realDrawElementsInstancedBaseVertex(mode, count, type, indices, instances, baseVertex); });
	}
	void APIENTRY DrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances, GLint baseVertex, GLuint baseInstance) {
		Draw(mode, [&]() { realDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instances, baseVertex, baseInstance); });
	}

	template<typename T>
	void* Wrap(void* real, T& realPtr, T wrapper) {
		if (real == nullptr)
			return nullptr;

		realPtr = reinterpret_cast<T>(real);
		return reinterpret_cast<void*>(wrapper);
	}

	void* GetProcAddress(const char* name, void* real) {
		#define WRAP(fn) if (strcmp(name, "gl" #fn) == 0) return Wrap(real, real##fn, &fn)
		WRAP(LineWidth);
		WRAP(DrawArrays);
		WRAP(DrawElements);
		WRAP(DrawRangeElements);
		WRAP(DrawArraysInstanced);
		WRAP(DrawElementsInstanced);
		WRAP(DrawArraysInstancedBaseInstance);
		WRAP(DrawElementsInstancedBaseInstance);
		WRAP(DrawElementsBaseVertex);
		WRAP(DrawElementsInstancedBaseVertex);
		WRAP(DrawElementsInstancedBaseVertexBaseInstance);
		#undef WRAP
		return real;
	}
}

// Keep legacy screen coordinates in logical points. Only the final default
// framebuffer is scaled; shadows, effects and Lua render-to-texture stay at
// their requested sizes. The world is composited before the high-density UI.
namespace Retina {
	PFNGLBINDFRAMEBUFFERPROC bindFramebuffer = nullptr;
	PFNGLGETINTEGERVPROC getIntegerv = nullptr;
	PFNGLVIEWPORTPROC viewport = nullptr;
	PFNGLSCISSORPROC scissor = nullptr;
	PFNGLBLITFRAMEBUFFERPROC blitFramebuffer = nullptr;
	PFNGLCOPYTEXSUBIMAGE2DPROC copyTexSubImage2D = nullptr;
	PFNGLREADPIXELSPROC readPixels = nullptr;
	PFNGLREADBUFFERPROC readBuffer = nullptr;
	PFNGLDRAWBUFFERPROC drawBuffer = nullptr;
	PFNGLDRAWBUFFERSPROC drawBuffers = nullptr;

	GLuint sceneFBO = 0;
	GLuint sceneColor = 0;
	GLuint sceneDepth = 0;
	int sceneX = 0, sceneY = 0;
	bool sceneHDR = false;
	GLuint hdrFBO = 0, hdrColor = 0, hdrDepth = 0;
	int hdrX = 0, hdrY = 0;
	GLuint logicalReadFBO = 0, logicalReadColor = 0;
	int logicalReadX = 0, logicalReadY = 0;
	GLint vertexClamp = GL_TRUE, fragmentClamp = GL_FIXED_ONLY;
	int logicalX = 1, logicalY = 1;
	float scaleX = 1.0f, scaleY = 1.0f;
	bool drawingScene = false;
	GLuint upscaleFBO = 0, upscaleColor = 0;
	int upscaleX = 0, upscaleY = 0;
	std::vector<uint8_t> scenePixels;
	bool metalFXFailed = false;
	bool sharedUpscaleLogged = false, cpuUpscaleLogged = false;

	bool HighDPI() { return retinaEnabled && (scaleX > 1.0f || scaleY > 1.0f); }

	bool Scaled(GLenum binding) {
		if (!HighDPI() || getIntegerv == nullptr)
			return false;
		GLint fbo = 0;
		getIntegerv(binding, &fbo);
		if (drawingScene) return hdrFrame && retinaMode == 2 && GLuint(fbo) == sceneFBO;
		return fbo == 0 || (hdrFrame && GLuint(fbo) == hdrFBO);
	}

	void Viewport(GLint x, GLint y, GLsizei w, GLsizei h) {
		if (Scaled(GL_DRAW_FRAMEBUFFER_BINDING)) {
			x = std::lround(x * scaleX); y = std::lround(y * scaleY);
			w = std::lround(w * scaleX); h = std::lround(h * scaleY);
		}
		viewport(x, y, w, h);
	}

	void Scissor(GLint x, GLint y, GLsizei w, GLsizei h) {
		if (Scaled(GL_DRAW_FRAMEBUFFER_BINDING)) {
			x = std::lround(x * scaleX); y = std::lround(y * scaleY);
			w = std::lround(w * scaleX); h = std::lround(h * scaleY);
		}
		scissor(x, y, w, h);
	}

	void GetIntegerv(GLenum name, GLint* values) {
		getIntegerv(name, values);
		if (hdrFrame && !drawingScene) {
			if ((name == GL_READ_FRAMEBUFFER_BINDING || name == GL_DRAW_FRAMEBUFFER_BINDING) && GLuint(*values) == hdrFBO)
				*values = 0;
			if (name == GL_READ_BUFFER || name == GL_DRAW_BUFFER || name == GL_DRAW_BUFFER0) {
				GLint fbo = 0;
				getIntegerv(name == GL_READ_BUFFER ? GL_READ_FRAMEBUFFER_BINDING : GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
				if (GLuint(fbo) == hdrFBO && *values == GL_COLOR_ATTACHMENT0) *values = GL_BACK;
			}
		}
		if ((name == GL_VIEWPORT || name == GL_SCISSOR_BOX) && Scaled(GL_DRAW_FRAMEBUFFER_BINDING)) {
			values[0] = std::lround(values[0] / scaleX); values[1] = std::lround(values[1] / scaleY);
			values[2] = std::lround(values[2] / scaleX); values[3] = std::lround(values[3] / scaleY);
		}
	}

	void BindFramebuffer(GLenum target, GLuint fbo) {
		if (fbo == 0) {
			if (drawingScene && sceneFBO) fbo = sceneFBO;
			else if (hdrFrame) fbo = hdrFBO;
		}
		bindFramebuffer(target, fbo);
	}

	GLenum ColorBuffer(GLenum buffer, GLenum binding) {
		if (!hdrFrame || (buffer != GL_BACK && buffer != GL_FRONT && buffer != GL_BACK_LEFT && buffer != GL_FRONT_LEFT))
			return buffer;
		GLint fbo = 0;
		getIntegerv(binding, &fbo);
		if (GLuint(fbo) == hdrFBO || GLuint(fbo) == sceneFBO)
			return GL_COLOR_ATTACHMENT0;
		return buffer;
	}

	void ReadBuffer(GLenum buffer) { readBuffer(ColorBuffer(buffer, GL_READ_FRAMEBUFFER_BINDING)); }
	void DrawBuffer(GLenum buffer) { drawBuffer(ColorBuffer(buffer, GL_DRAW_FRAMEBUFFER_BINDING)); }
	void DrawBuffers(GLsizei count, const GLenum* buffers) {
		if (!hdrFrame || count <= 0 || !buffers) { drawBuffers(count, buffers); return; }
		bool needsMapping = false;
		for (GLsizei i = 0; i < count; ++i)
			needsMapping |= buffers[i] == GL_BACK || buffers[i] == GL_FRONT || buffers[i] == GL_BACK_LEFT || buffers[i] == GL_FRONT_LEFT;
		if (!needsMapping) { drawBuffers(count, buffers); return; }
		std::vector<GLenum> mapped(buffers, buffers + count);
		for (GLenum& buffer: mapped) buffer = ColorBuffer(buffer, GL_DRAW_FRAMEBUFFER_BINDING);
		drawBuffers(count, mapped.data());
	}

	void DestroyHDR() {
		if (hdrFBO || logicalReadFBO) MacMetalFX::WaitShared();
		if (logicalReadFBO) glDeleteFramebuffers(1, &logicalReadFBO);
		if (logicalReadColor) glDeleteRenderbuffers(1, &logicalReadColor);
		logicalReadFBO = logicalReadColor = 0;
		logicalReadX = logicalReadY = 0;
		if (hdrFBO) glDeleteFramebuffers(1, &hdrFBO);
		if (hdrColor) glDeleteTextures(1, &hdrColor);
		if (hdrDepth) glDeleteRenderbuffers(1, &hdrDepth);
		hdrFBO = hdrColor = hdrDepth = 0;
		hdrX = hdrY = 0;
	}

	bool EnsureHDR() {
		if (hdrFBO && hdrX == surfSizeX && hdrY == surfSizeY) return true;
		GLint read = 0, draw = 0, rbo = 0, tex = 0, unpack = 0;
		getIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read);
		getIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw);
		getIntegerv(GL_RENDERBUFFER_BINDING, &rbo);
		getIntegerv(GL_TEXTURE_BINDING_2D, &tex);
		getIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &unpack);
		DestroyHDR();
		glGenFramebuffers(1, &hdrFBO);
		glGenTextures(1, &hdrColor);
		glBindTexture(GL_TEXTURE_2D, hdrColor);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, surfSizeX, surfSizeY, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		bindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdrColor, 0);
		glGenRenderbuffers(1, &hdrDepth);
		glBindRenderbuffer(GL_RENDERBUFFER, hdrDepth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, surfSizeX, surfSizeY);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, hdrDepth);
		const bool valid = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
		if (valid) {
			// Establish defined GL contents before Metal writes this allocation.
			// Otherwise Zink may discard the external writes when UI first draws.
			const bool clipped = glIsEnabled(GL_SCISSOR_TEST);
			GLboolean mask[4];
			glGetBooleanv(GL_COLOR_WRITEMASK, mask);
			glDisable(GL_SCISSOR_TEST);
			glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
			const GLfloat black[4] = {};
			glClearBufferfv(GL_COLOR, 0, black);
			glColorMask(mask[0], mask[1], mask[2], mask[3]);
			if (clipped) glEnable(GL_SCISSOR_TEST);
		}
		bindFramebuffer(GL_READ_FRAMEBUFFER, read);
		bindFramebuffer(GL_DRAW_FRAMEBUFFER, draw);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo);
		glBindTexture(GL_TEXTURE_2D, tex);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpack);
		if (!valid) { DestroyHDR(); return false; }
		hdrX = surfSizeX; hdrY = surfSizeY;
		return true;
	}

	bool RenderHDRScene() {
		const bool spatial = retinaMode == 1 && HighDPI() && configHandler->GetBool("MacMetalFXSpatial");
		bool scaled = false;
		if (!spatial) sharedUpscaleFailed = false;
		if (spatial && !sharedUpscaleFailed) {
			void* device = nullptr;
			void* source = nullptr;
			void* outputDevice = nullptr;
			void* output = nullptr;
			bindFramebuffer(GL_READ_FRAMEBUFFER, sceneFBO);
			glFinish();
			const bool sourceReady = GetSharedColorBuffer(device, source);
			bindFramebuffer(GL_READ_FRAMEBUFFER, hdrFBO);
			const bool outputReady = GetSharedColorBuffer(outputDevice, output);
			scaled = sourceReady && outputReady && device == outputDevice &&
				MacMetalFX::UpscaleShared(device, source, output);
			if (!scaled) {
				sharedUpscaleFailed = true;
				LOG_L(L_WARNING, "[MacHDR] MetalFX unavailable (%s); using floating-point bilinear scaling",
					sourceReady && outputReady ? MacMetalFX::GetError() : "Mesa export unavailable");
			}
		}
		bindFramebuffer(GL_READ_FRAMEBUFFER, sceneFBO);
		bindFramebuffer(GL_DRAW_FRAMEBUFFER, hdrFBO);
		if (!scaled)
			blitFramebuffer(0, 0, sceneX, sceneY, 0, 0, surfSizeX, surfSizeY, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		if (!hdrLogged) LOG("[MacHDR] floating-point game lighting -> HDR display; no added bloom or color grading");
		hdrLogged = true;
		return true;
	}

	void BlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask, GLenum filter) {
		if (Scaled(GL_READ_FRAMEBUFFER_BINDING)) {
			sx0 = std::lround(sx0 * scaleX); sx1 = std::lround(sx1 * scaleX);
			sy0 = std::lround(sy0 * scaleY); sy1 = std::lround(sy1 * scaleY);
		}
		if (Scaled(GL_DRAW_FRAMEBUFFER_BINDING)) {
			dx0 = std::lround(dx0 * scaleX); dx1 = std::lround(dx1 * scaleX);
			dy0 = std::lround(dy0 * scaleY); dy1 = std::lround(dy1 * scaleY);
		}
		blitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
	}

	void DestroyScene() {
		MacMetalFX::WaitShared();
		if (sceneFBO != 0) glDeleteFramebuffers(1, &sceneFBO);
		if (sceneColor != 0) glDeleteRenderbuffers(1, &sceneColor);
		if (sceneDepth != 0) glDeleteRenderbuffers(1, &sceneDepth);
		sceneFBO = sceneColor = sceneDepth = 0;
		sceneX = sceneY = 0;
	}

	void DestroyUpscaler() {
		MacMetalFX::WaitShared();
		if (upscaleFBO != 0) glDeleteFramebuffers(1, &upscaleFBO);
		if (upscaleColor != 0) glDeleteTextures(1, &upscaleColor);
		upscaleFBO = upscaleColor = 0;
		upscaleX = upscaleY = 0;
		scenePixels.clear();
		metalFXFailed = false;
		sharedUpscaleFailed = sharedPresentFailed = false;
		sharedUpscaleLogged = cpuUpscaleLogged = sharedPresentLogged = false;
		presentedSharedLastFrame = false;
		MacMetalFX::Destroy();
	}

	bool EnsureUpscaleTarget() {
		GLint previousTexture = 0, unpackBuffer = 0;
		getIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture);
		getIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &unpackBuffer);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		if (upscaleFBO == 0) {
			glGenFramebuffers(1, &upscaleFBO);
			glGenTextures(1, &upscaleColor);
		}
		glBindTexture(GL_TEXTURE_2D, upscaleColor);
		bool valid = true;
		if (upscaleX != surfSizeX || upscaleY != surfSizeY) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, surfSizeX, surfSizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			bindFramebuffer(GL_READ_FRAMEBUFFER, upscaleFBO);
			glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, upscaleColor, 0);
			valid = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
			if (valid) {
				upscaleX = surfSizeX; upscaleY = surfSizeY;
				sharedUpscaleLogged = cpuUpscaleLogged = false;
			}
		}
		bindFramebuffer(GL_READ_FRAMEBUFFER, sceneFBO);
		glBindTexture(GL_TEXTURE_2D, previousTexture);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpackBuffer);
		return valid;
	}

	// Called with the scene bound for reading, default framebuffer for drawing,
	// and scissor disabled. Keep the Metal boundary here so UI is never upscaled.
	bool UpscaleScene() {
		if (!configHandler->GetBool("MacMetalFXSpatial")) {
			metalFXFailed = false;
			return false;
		}
		if (metalFXFailed)
			return false;
		if (!EnsureUpscaleTarget())
			return false;

		if (UseSharedGPU() && !sharedUpscaleFailed) {
			bool success = false;
			const char* reason = "bundled Mesa export interface unavailable";
			if (SharedExportsAvailable()) {
				glFlush();
				void* device = nullptr;
				void* source = nullptr;
				void* nativeQueue = nullptr;
				void* outputDevice = nullptr;
				void* output = nullptr;
				const bool sourceReady = GetSharedColorBuffer(device, source, &nativeQueue);
				bindFramebuffer(GL_READ_FRAMEBUFFER, upscaleFBO);
				const bool outputReady = GetSharedColorBuffer(outputDevice, output);
				reason = "scene or output framebuffer is not shareable";
				if (sourceReady && outputReady && device == outputDevice) {
					success = MacMetalFX::UpscaleShared(device, source, output, SubmitSharedGPU(nativeQueue));
					reason = MacMetalFX::GetError();
				}
				if (success) {
					blitFramebuffer(0, 0, surfSizeX, surfSizeY, 0, 0, surfSizeX, surfSizeY, GL_COLOR_BUFFER_BIT, GL_NEAREST);
				}
				bindFramebuffer(GL_READ_FRAMEBUFFER, sceneFBO);
			}
			if (success) {
				if (!sharedUpscaleLogged)
					LOG("[MacMetalFX] spatial upscale %dx%d -> %dx%d; shared GPU textures, native-resolution UI", logicalX, logicalY, surfSizeX, surfSizeY);
				sharedUpscaleLogged = true;
				return true;
			}
			sharedUpscaleFailed = true;
			LOG_L(L_WARNING, "[MacGPUSharing] MetalFX sharing unavailable (%s); using CPU staging", reason);
			MacMetalFX::Destroy();
		}

		// Widgets may leave PBOs and pixel-store offsets bound. Neither transfer
		// may inherit those, and all bindings must be restored before UI draws.
		constexpr GLenum packNames[] = {GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS};
		constexpr GLenum unpackNames[] = {GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS};
		GLint pack[4], unpack[4], packBuffer = 0, unpackBuffer = 0, previousTexture = 0;
		getIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuffer);
		getIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &unpackBuffer);
		getIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture);
		for (int i = 0; i < 4; ++i) {
			getIntegerv(packNames[i], &pack[i]);
			getIntegerv(unpackNames[i], &unpack[i]);
			glPixelStorei(packNames[i], i == 0 ? 4 : 0);
			glPixelStorei(unpackNames[i], i == 0 ? 4 : 0);
		}
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		scenePixels.resize(size_t(logicalX) * logicalY * 4);
		readPixels(0, 0, logicalX, logicalY, GL_RGBA, GL_UNSIGNED_BYTE, scenePixels.data());
		const void* output = nullptr;
		int outputRowLength = 0;
		bool success = MacMetalFX::Upscale(scenePixels.data(), logicalX, logicalY, surfSizeX, surfSizeY, output, outputRowLength);
		if (success) {
			glBindTexture(GL_TEXTURE_2D, upscaleColor);
			glPixelStorei(GL_UNPACK_ROW_LENGTH, outputRowLength);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, surfSizeX, surfSizeY, GL_RGBA, GL_UNSIGNED_BYTE, output);
			bindFramebuffer(GL_READ_FRAMEBUFFER, upscaleFBO);
			blitFramebuffer(0, 0, surfSizeX, surfSizeY, 0, 0, surfSizeX, surfSizeY, GL_COLOR_BUFFER_BIT, GL_NEAREST);
			if (!cpuUpscaleLogged)
				LOG("[MacMetalFX] spatial upscale %dx%d -> %dx%d; CPU staging, native-resolution UI", logicalX, logicalY, surfSizeX, surfSizeY);
			cpuUpscaleLogged = true;
		}
		bindFramebuffer(GL_READ_FRAMEBUFFER, sceneFBO);
		glBindTexture(GL_TEXTURE_2D, previousTexture);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, packBuffer);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpackBuffer);
		for (int i = 0; i < 4; ++i) {
			glPixelStorei(packNames[i], pack[i]);
			glPixelStorei(unpackNames[i], unpack[i]);
		}
		if (!success) {
			metalFXFailed = true;
			LOG_L(L_WARNING, "[MacMetalFX] %s; using bilinear scaling (toggle off/on to retry)", MacMetalFX::GetError());
			MacMetalFX::Destroy();
		}
		return success;
	}

	bool EnsureScene() {
		const int width = hdrFrame && retinaMode == 2 ? surfSizeX : logicalX;
		const int height = hdrFrame && retinaMode == 2 ? surfSizeY : logicalY;
		if (sceneFBO != 0 && sceneX == width && sceneY == height && sceneHDR == hdrFrame)
			return true;
		GLint read = 0, draw = 0, rbo = 0;
		getIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read);
		getIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw);
		getIntegerv(GL_RENDERBUFFER_BINDING, &rbo);
		DestroyScene();
		glGenFramebuffers(1, &sceneFBO);
		bindFramebuffer(GL_FRAMEBUFFER, sceneFBO);
		glGenRenderbuffers(1, &sceneColor);
		glBindRenderbuffer(GL_RENDERBUFFER, sceneColor);
		glRenderbufferStorage(GL_RENDERBUFFER, hdrFrame ? GL_RGBA16F : GL_RGBA8, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, sceneColor);
		glGenRenderbuffers(1, &sceneDepth);
		glBindRenderbuffer(GL_RENDERBUFFER, sceneDepth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, sceneDepth);
		const bool valid = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
		bindFramebuffer(GL_READ_FRAMEBUFFER, read);
		bindFramebuffer(GL_DRAW_FRAMEBUFFER, draw);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo);
		if (!valid) {
			DestroyScene();
			LOG_L(L_ERROR, "[MacRetina] could not create scene framebuffer");
			return false;
		}
		sceneX = width; sceneY = height; sceneHDR = hdrFrame;
		return true;
	}

	bool EnsureLogicalReadTarget() {
		if (logicalReadFBO && logicalReadX == logicalX && logicalReadY == logicalY) return true;
		GLint read = 0, draw = 0, rbo = 0;
		getIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read);
		getIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw);
		getIntegerv(GL_RENDERBUFFER_BINDING, &rbo);
		if (!logicalReadFBO) {
			glGenFramebuffers(1, &logicalReadFBO);
			glGenRenderbuffers(1, &logicalReadColor);
		}
		bindFramebuffer(GL_FRAMEBUFFER, logicalReadFBO);
		glBindRenderbuffer(GL_RENDERBUFFER, logicalReadColor);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA16F, logicalX, logicalY);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, logicalReadColor);
		const bool valid = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
		bindFramebuffer(GL_READ_FRAMEBUFFER, read);
		bindFramebuffer(GL_DRAW_FRAMEBUFFER, draw);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo);
		if (valid) { logicalReadX = logicalX; logicalReadY = logicalY; }
		return valid;
	}

	template<typename F>
	void ReadLogical(F&& read) {
		if (!Scaled(GL_READ_FRAMEBUFFER_BINDING)) {
			read();
			return;
		}
		GLint previousRead = 0, previousDraw = 0;
		getIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousRead);
		getIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDraw);
		// Full Retina HDR can be copied by an effect while the world is drawing.
		// Downsample into a separate target; an overlapping self-blit is undefined.
		const bool copyingWorld = hdrFrame && drawingScene;
		if (!(copyingWorld ? EnsureLogicalReadTarget() : EnsureScene())) {
			read();
			return;
		}
		const GLuint target = copyingWorld ? logicalReadFBO : sceneFBO;
		const bool clipped = glIsEnabled(GL_SCISSOR_TEST);
		glDisable(GL_SCISSOR_TEST);
		bindFramebuffer(GL_DRAW_FRAMEBUFFER, target);
		blitFramebuffer(0, 0, surfSizeX, surfSizeY, 0, 0, logicalX, logicalY, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		bindFramebuffer(GL_READ_FRAMEBUFFER, target);
		bindFramebuffer(GL_DRAW_FRAMEBUFFER, previousDraw);
		if (clipped) glEnable(GL_SCISSOR_TEST);
		read();
		bindFramebuffer(GL_READ_FRAMEBUFFER, previousRead);
	}

	void CopyTexSubImage2D(GLenum target, GLint level, GLint xoff, GLint yoff, GLint x, GLint y, GLsizei w, GLsizei h) {
		ReadLogical([&]() { copyTexSubImage2D(target, level, xoff, yoff, x, y, w, h); });
	}

	void ReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void* pixels) {
		ReadLogical([&]() { readPixels(x, y, w, h, format, type, pixels); });
	}

	void* GetProcAddress(const char* name, void* real) {
		#define WRAP(glName, pointer, wrapper) if (strcmp(name, glName) == 0 && real != nullptr) { pointer = reinterpret_cast<decltype(pointer)>(real); return reinterpret_cast<void*>(&wrapper); }
		WRAP("glBindFramebuffer", bindFramebuffer, BindFramebuffer);
		WRAP("glBindFramebufferEXT", bindFramebuffer, BindFramebuffer);
		WRAP("glGetIntegerv", getIntegerv, GetIntegerv);
		WRAP("glViewport", viewport, Viewport);
		WRAP("glScissor", scissor, Scissor);
		WRAP("glBlitFramebuffer", blitFramebuffer, BlitFramebuffer);
		WRAP("glBlitFramebufferEXT", blitFramebuffer, BlitFramebuffer);
		WRAP("glCopyTexSubImage2D", copyTexSubImage2D, CopyTexSubImage2D);
		WRAP("glReadPixels", readPixels, ReadPixels);
		WRAP("glReadBuffer", readBuffer, ReadBuffer);
		WRAP("glDrawBuffer", drawBuffer, DrawBuffer);
		WRAP("glDrawBuffers", drawBuffers, DrawBuffers);
		#undef WRAP
		return real;
	}

	void UpdateSurface() {
		int wx = 0, wy = 0, px = 0, py = 0;
		SDL_GetWindowSize(presenterWindow, &wx, &wy);
		SDL_GetRendererOutputSize(renderer, &px, &py);
		if (wx <= 0 || wy <= 0 || px <= 0 || py <= 0)
			return;
		const bool sameLogicalSize = logicalX == wx && logicalY == wy;
		const float oldScaleX = scaleX, oldScaleY = scaleY;
		logicalX = wx; logicalY = wy;
		if (!retinaEnabled) { px = wx; py = wy; }
		scaleX = float(px) / wx; scaleY = float(py) / wy;
		if (px != surfSizeX || py != surfSizeY) {
			std::lock_guard<std::mutex> lck(surfMutex);
			CreateSurface(px, py);
			LOG("[MacRetina] mode=%d logical=%dx%d backing=%dx%d scene=%dx%d density=%.2fx%.2f", retinaMode, wx, wy, px, py, retinaMode == 2 ? px : wx, retinaMode == 2 ? py : wy, scaleX, scaleY);
			// Moving to a display with a different backing scale need not resize
			// the logical window. Rebuild UI lists so their cached clips match.
			if (sameLogicalSize && (oldScaleX != scaleX || oldScaleY != scaleY)) {
				SDL_Event event = {};
				event.type = SDL_WINDOWEVENT;
				event.window.windowID = SDL_GetWindowID(presenterWindow);
				event.window.event = SDL_WINDOWEVENT_SIZE_CHANGED;
				event.window.data1 = wx; event.window.data2 = wy;
				SDL_PushEvent(&event);
			}
		}
	}
}

int MacEGLBridge::GetFontRasterScale() { return retinaEnabled ? 2 : 1; }
int MacEGLBridge::GetRetinaMode() { return retinaMode; }
void MacEGLBridge::ScaleSceneSize(int& width, int& height)
{
	if (retinaMode != 2) return;
	width = std::lround(width * Retina::scaleX);
	height = std::lround(height * Retina::scaleY);
}

void MacEGLBridge::CopyToTexturePixels(unsigned int target, int level, int xoff, int yoff, int x, int y, int width, int height)
{
	Retina::copyTexSubImage2D(target, level, xoff, yoff, x, y, width, height);
}

void MacEGLBridge::BeginScene()
{
	if (context == nullptr || renderer == nullptr) return;
	Retina::UpdateSurface();
	// Keep the completed float frame readable between draws (screenshots and
	// widgets run during Update), then detach before reallocating its resources.
	Retina::bindFramebuffer(GL_FRAMEBUFFER, 0);
	hdrFrame = false;
	hdrSceneDrawn = true;
	const bool requested = configHandler->GetBool("MacHDR");
	if (requested != previousHDRRequest) {
		hdrFailed = hdrLogged = hdrUnavailableLogged = false;
		loggedHDRHeadroom = 0.0f;
		sharedPresentFailed = false;
		previousHDRRequest = requested;
	}
	if (!requested) { hdrFailed = hdrLogged = false; Retina::DestroyHDR(); }
	hdrFrame = requested && !hdrFailed && SharedExportsAvailable() &&
		MacMetalFX::ConfigureHDR(SDL_RenderGetMetalLayer(renderer), NativeWindow(), true);
	if (requested && !hdrFailed && !hdrFrame && !hdrUnavailableLogged) {
		LOG_L(L_WARNING, "[MacHDR] HDR display or bundled Mesa sharing unavailable; using SDR");
		hdrUnavailableLogged = true;
	}
	if (hdrFrame) hdrUnavailableLogged = false;
	if (hdrFrame && !Retina::EnsureHDR()) {
		LOG_L(L_WARNING, "[MacHDR] floating-point composition target unavailable; using SDR");
		hdrFrame = false; hdrFailed = true;
	}
	Retina::drawingScene = (hdrFrame || (retinaMode == 1 && Retina::HighDPI())) && Retina::EnsureScene();
	if (hdrFrame && !Retina::drawingScene) { hdrFrame = false; hdrFailed = true; }
	if (hdrFrame) {
		Retina::getIntegerv(GL_CLAMP_VERTEX_COLOR, &Retina::vertexClamp);
		Retina::getIntegerv(GL_CLAMP_FRAGMENT_COLOR, &Retina::fragmentClamp);
		// Legacy vertex-only shaders (notably the mirrored map edge) rely on
		// clamped gl_FrontColor alpha. HDR energy comes from fragment shaders
		// and additive blending, which remain unclamped in the float target.
		glClampColor(GL_CLAMP_VERTEX_COLOR, GL_TRUE);
		glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	} else {
		MacMetalFX::ConfigureHDR(SDL_RenderGetMetalLayer(renderer), NativeWindow(), false);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void MacEGLBridge::BeginUI()
{
	if (context == nullptr || renderer == nullptr) return;
	Retina::UpdateSurface();
	if (Retina::drawingScene) {
		const bool clipped = glIsEnabled(GL_SCISSOR_TEST);
		glDisable(GL_SCISSOR_TEST);
		Retina::bindFramebuffer(GL_READ_FRAMEBUFFER, Retina::sceneFBO);
		const bool wasHDR = hdrFrame;
		if (hdrFrame && !Retina::RenderHDRScene()) {
			hdrFrame = false;
			MacMetalFX::ConfigureHDR(SDL_RenderGetMetalLayer(renderer), NativeWindow(), false);
		}
		Retina::bindFramebuffer(GL_DRAW_FRAMEBUFFER, hdrFrame ? Retina::hdrFBO : 0);
		if (!hdrFrame && (wasHDR || !Retina::UpscaleScene()))
			Retina::blitFramebuffer(0, 0, Retina::sceneX, Retina::sceneY, 0, 0, surfSizeX, surfSizeY, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		Retina::blitFramebuffer(0, 0, Retina::sceneX, Retina::sceneY, 0, 0, surfSizeX, surfSizeY, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
		if (wasHDR) {
			glClampColor(GL_CLAMP_VERTEX_COLOR, Retina::vertexClamp);
			glClampColor(GL_CLAMP_FRAGMENT_COLOR, Retina::fragmentClamp);
		}
		if (clipped) glEnable(GL_SCISSOR_TEST);
	}
	Retina::drawingScene = false;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// World and offscreen passes can leave a smaller viewport. UI always starts
	// with logical screen bounds, then the GL wrappers apply the backing scale.
	glViewport(globalRendering->viewPosX, globalRendering->viewPosY, globalRendering->viewSizeX, globalRendering->viewSizeY);
	glScissor(globalRendering->viewPosX, globalRendering->viewPosY, globalRendering->viewSizeX, globalRendering->viewSizeY);
}


bool MacEGLBridge::Active() { return (context != nullptr); }


void* MacEGLBridge::CreateContext(SDL_Window* window, int major, int minor, bool core)
{
	presenterWindow = window;
	retinaMode = configHandler->GetInt("MacRetinaMode");
	retinaEnabled = retinaMode != 0;
	SetupEnvironment();

	if (!LoadEGL())
		return nullptr;

	if ((display = egl.GetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, nullptr, nullptr)) == nullptr) {
		LOG_L(L_ERROR, "[MacEGLBridge] eglGetPlatformDisplayEXT(surfaceless) failed: 0x%x", egl.GetError());
		return nullptr;
	}

	EGLint vmaj = 0, vmin = 0;
	if (!egl.Initialize(display, &vmaj, &vmin)) {
		LOG_L(L_ERROR, "[MacEGLBridge] eglInitialize failed: 0x%x", egl.GetError());
		return nullptr;
	}
	LOG("[MacEGLBridge] EGL %d.%d initialized (GALLIUM_DRIVER=%s VK_DRIVER_FILES=%s)", vmaj, vmin, getenv("GALLIUM_DRIVER"), getenv("VK_DRIVER_FILES"));

	egl.BindAPI(EGL_OPENGL_API);

	const EGLint cfgAttrs[] = {
		EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
		EGL_NONE
	};
	EGLint numCfgs = 0;
	if (!egl.ChooseConfig(display, cfgAttrs, &config, 1, &numCfgs) || numCfgs < 1) {
		LOG_L(L_ERROR, "[MacEGLBridge] eglChooseConfig failed: 0x%x", egl.GetError());
		return nullptr;
	}

	const EGLint ctxAttrs[] = {
		EGL_CONTEXT_MAJOR_VERSION, major,
		EGL_CONTEXT_MINOR_VERSION, minor,
		EGL_CONTEXT_OPENGL_PROFILE_MASK, core ? EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT : EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
		EGL_NONE
	};
	if ((context = egl.CreateContext(display, config, nullptr, ctxAttrs)) == nullptr) {
		LOG_L(L_ERROR, "[MacEGLBridge] GL %d.%d %s context failed: 0x%x", major, minor, core ? "core" : "compatibility", egl.GetError());
		return nullptr;
	}

	int wx = 0, wy = 0;
	SDL_GetWindowSize(window, &wx, &wy);
	if (!CreateSurface(wx, wy) || !egl.MakeCurrent(display, surface, surface, context)) {
		LOG_L(L_ERROR, "[MacEGLBridge] eglMakeCurrent failed: 0x%x", egl.GetError());
		return nullptr;
	}

	if (!CreatePresenter(window))
		return nullptr;

	return context;
}

void MacEGLBridge::MakeCurrent(bool clear)
{
	if (context == nullptr)
		return;

	std::lock_guard<std::mutex> lck(surfMutex);
	if (clear)
		egl.MakeCurrent(display, nullptr, nullptr, nullptr);
	else
		egl.MakeCurrent(display, surface, surface, context);
}

void MacEGLBridge::DestroyContext()
{
	if (context == nullptr)
		return;
	MacMetalFX::StopDisplayLink();
	MacMetalFX::WaitShared();

	if (texture != nullptr)
		SDL_DestroyTexture(texture);
	if (renderer != nullptr)
		SDL_DestroyRenderer(renderer);

	texture = nullptr;
	renderer = nullptr;
	texSizeX = texSizeY = 0;

	egl.MakeCurrent(display, surface, surface, context);
	Retina::drawingScene = false;
	Retina::DestroyScene();
	Retina::DestroyHDR();
	Retina::DestroyUpscaler();
	for (unsigned int i = 0; i < NUM_PBOS; i++) {
		if (fences[i] != nullptr)
			glDeleteSync(fences[i]);
		fences[i] = nullptr;
	}
	if (pbos[0] != 0)
		glDeleteBuffers(NUM_PBOS, pbos);
	std::fill(std::begin(pbos), std::end(pbos), 0);

	egl.MakeCurrent(display, nullptr, nullptr, nullptr);
	egl.DestroySurface(display, surface);
	egl.DestroyContext(display, context);
	egl.Terminate(display);

	surface = nullptr;
	context = nullptr;
	display = nullptr;
}

void* MacEGLBridge::GetProcAddress(const char* name)
{
	if (!LoadEGL())
		return nullptr;

	return Retina::GetProcAddress(name, WideLines::GetProcAddress(name, egl.GetProcAddress(name)));
}

void MacEGLBridge::SetSwapInterval(int interval)
{
	if (renderer != nullptr && SDL_RenderSetVSync(renderer, 1) != 0)
		LOG_L(L_ERROR, "[MacEGLBridge] could not enable required VSync: %s", SDL_GetError());
}

void MacEGLBridge::WaitForFrame()
{
	if (context == nullptr || renderer == nullptr)
		return;
	if (!configHandler->GetBool("MacDisplayLink") || pixelPresentFailed) {
		MacMetalFX::StopDisplayLink();
		displayLinkLogged = false;
		return;
	}
	SDL_DisplayMode mode = {};
	SDL_GetCurrentDisplayMode(SDL_GetWindowDisplayIndex(presenterWindow), &mode);
	const bool visible = !(SDL_GetWindowFlags(presenterWindow) & (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED));
	if (MacMetalFX::WaitForFrame(SDL_RenderGetMetalLayer(renderer), mode.refresh_rate, visible) && !displayLinkLogged) {
		LOG("[MacDisplayLink] display-driven frame pacing enabled; preferred latency 2 frames, display %d Hz", mode.refresh_rate);
		displayLinkLogged = true;
	}
}

void MacEGLBridge::SwapWindow(SDL_Window* window)
{
	if (context == nullptr || renderer == nullptr)
		return;

	// A controller that did not draw a world frame (lobby/loading) uses SDR.
	if (!hdrSceneDrawn && hdrFrame) {
		hdrFrame = false;
		Retina::bindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	hdrSceneDrawn = false;
	if (!hdrFrame) MacMetalFX::ConfigureHDR(SDL_RenderGetMetalLayer(renderer), NativeWindow(), false);
	if (UseSharedGPU() && !sharedPresentFailed) {
		GLint previousReadFBO = 0;
		Retina::getIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFBO);
		Retina::bindFramebuffer(GL_READ_FRAMEBUFFER, hdrFrame ? Retina::hdrFBO : 0);
		GLint previousReadBuffer = GL_BACK;
		Retina::getIntegerv(GL_READ_BUFFER, &previousReadBuffer);
		glReadBuffer(hdrFrame ? GL_COLOR_ATTACHMENT0 : GL_BACK);
		bool success = false;
		const char* reason = "bundled Mesa export interface unavailable";
		if (SharedExportsAvailable()) {
			glFlush();
			void* device = nullptr;
			void* source = nullptr;
			void* nativeQueue = nullptr;
			reason = "presentation framebuffer is not shareable";
			if (GetSharedColorBuffer(device, source, &nativeQueue)) {
				const float headroom = hdrFrame ? MacMetalFX::HDRHeadroom(NativeWindow()) : 0.0f;
				if (hdrFrame && std::abs(headroom - loggedHDRHeadroom) >= 0.25f) {
					LOG("[MacHDR] display brightness headroom %.2fx SDR white", headroom);
					loggedHDRHeadroom = headroom;
				}
				success = MacMetalFX::PresentShared(device, source, SDL_RenderGetMetalLayer(renderer), headroom, SubmitSharedGPU(nativeQueue));
				reason = MacMetalFX::GetError();
			}
		}
		glReadBuffer(previousReadBuffer);
		Retina::bindFramebuffer(GL_READ_FRAMEBUFFER, previousReadFBO);
		if (success) {
			if (!sharedPresentLogged)
				LOG("[MacGPUSharing] presenting shared GPU texture; no final-frame CPU readback");
			sharedPresentLogged = true;
			presentedSharedLastFrame = true;
			Retina::bindFramebuffer(GL_FRAMEBUFFER, hdrFrame ? Retina::hdrFBO : 0);
			Retina::UpdateSurface();
			return;
		}
		sharedPresentFailed = true;
		LOG_L(L_WARNING, "[MacGPUSharing] shared presentation unavailable (%s); using PBO readback", reason);
	}
	if (hdrFrame) {
		const bool clipped = glIsEnabled(GL_SCISSOR_TEST);
		glDisable(GL_SCISSOR_TEST);
		Retina::bindFramebuffer(GL_READ_FRAMEBUFFER, Retina::hdrFBO);
		Retina::bindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		Retina::blitFramebuffer(0, 0, surfSizeX, surfSizeY, 0, 0, surfSizeX, surfSizeY, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		if (clipped) glEnable(GL_SCISSOR_TEST);
		Retina::bindFramebuffer(GL_FRAMEBUFFER, 0);
		hdrFrame = false; hdrFailed = true;
		MacMetalFX::ConfigureHDR(SDL_RenderGetMetalLayer(renderer), NativeWindow(), false);
		LOG_L(L_WARNING, "[MacHDR] HDR presentation unavailable; using SDR (toggle HDR off/on to retry)");
	}
	if (presentedSharedLastFrame) {
		// Discard frames queued before sharing was enabled, rather than showing stale images.
		for (unsigned int i = 0; i < NUM_PBOS; ++i) {
			if (fences[i]) glDeleteSync(fences[i]);
			fences[i] = nullptr;
			pboSizes[i][0] = pboSizes[i][1] = 0;
		}
		if (texture) SDL_DestroyTexture(texture);
		texture = nullptr;
		texSizeX = texSizeY = 0;
		frameIdx = 0;
		presentedSharedLastFrame = false;
	}

	GLint prevPackBuf = 0, prevReadFBO = 0, prevAlign = 4, prevRowLen = 0, prevReadBuf = GL_BACK;
	glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &prevPackBuf);
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFBO);
	glGetIntegerv(GL_PACK_ALIGNMENT, &prevAlign);
	glGetIntegerv(GL_PACK_ROW_LENGTH, &prevRowLen);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glGetIntegerv(GL_READ_BUFFER, &prevReadBuf);
	glReadBuffer(GL_BACK);
	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glPixelStorei(GL_PACK_ROW_LENGTH, 0);

	if (pbos[0] == 0)
		glGenBuffers(NUM_PBOS, pbos);

	// asynchronous readback: queue this frame into one PBO and present the oldest one; the flush
	// submits the frame right away so mapping never has to wait for the frame currently in flight
	const unsigned int cur = frameIdx % NUM_PBOS;
	const unsigned int prv = (frameIdx + 1) % NUM_PBOS;

	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[cur]);
	if (pboSizes[cur][0] != surfSizeX || pboSizes[cur][1] != surfSizeY) {
		glBufferData(GL_PIXEL_PACK_BUFFER, surfSizeX * surfSizeY * 4, nullptr, GL_STREAM_READ);
		pboSizes[cur][0] = surfSizeX;
		pboSizes[cur][1] = surfSizeY;
	}
	// Presentation reads physical pixels; Lua's read/copy APIs retain logical coordinates.
	Retina::readPixels(0, 0, surfSizeX, surfSizeY, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	if (fences[cur] != nullptr)
		glDeleteSync(fences[cur]);
	fences[cur] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
	glFlush();

	const int px = pboSizes[prv][0];
	const int py = pboSizes[prv][1];
	bool presentedPixels = false;

	if (fences[prv] != nullptr && px > 0 && py > 0) {
		glClientWaitSync(fences[prv], 0, GL_TIMEOUT_IGNORED);
		glDeleteSync(fences[prv]);
		fences[prv] = nullptr;

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[prv]);
		const void* data = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, px * py * 4, GL_MAP_READ_BIT);

		if (data != nullptr) {
			if (!pixelPresentFailed) {
				presentedPixels = MacMetalFX::PresentPixels(data, px, py, SDL_RenderGetMetalLayer(renderer));
				if (presentedPixels && !pixelPresentLogged) {
					LOG("[MacEGLBridge] presenting PBO pixels through reusable Metal textures");
					pixelPresentLogged = true;
				} else if (!presentedPixels) {
					pixelPresentFailed = true;
					MacMetalFX::StopDisplayLink();
					MacMetalFX::WaitShared();
					LOG_L(L_WARNING, "[MacEGLBridge] reusable Metal presentation failed (%s); using SDL upload", MacMetalFX::GetError());
				}
			}
			if (!presentedPixels && (texture == nullptr || texSizeX != px || texSizeY != py)) {
				if (texture != nullptr)
					SDL_DestroyTexture(texture);
				texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, px, py);
				texSizeX = px;
				texSizeY = py;
			}

			void* pixels = nullptr;
			int pitch = 0;
			if (!presentedPixels && texture != nullptr && SDL_LockTexture(texture, nullptr, &pixels, &pitch) == 0) {
				if (pitch == px * 4) {
					memcpy(pixels, data, px * py * 4);
				} else {
					for (int y = 0; y < py; ++y)
						memcpy(static_cast<uint8_t*>(pixels) + y * pitch, static_cast<const uint8_t*>(data) + y * px * 4, px * 4);
				}
				SDL_UnlockTexture(texture);
			}

			glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
		}
	}

	glBindBuffer(GL_PIXEL_PACK_BUFFER, prevPackBuf);
	glReadBuffer(prevReadBuf);
	glPixelStorei(GL_PACK_ALIGNMENT, prevAlign);
	glPixelStorei(GL_PACK_ROW_LENGTH, prevRowLen);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFBO);

	if (!presentedPixels && texture != nullptr) {
		SDL_RenderCopyEx(renderer, texture, nullptr, nullptr, 0.0, nullptr, SDL_FLIP_VERTICAL);
		SDL_RenderPresent(renderer);
	}

	frameIdx++;

	Retina::UpdateSurface();
}

#endif
