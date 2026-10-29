/* This file is part of the Spring engine (GPL v2 or later), see LICENSE.html */
// MetalFX and macOS HDR integration assisted by OpenAI Codex.

#include "MacEGLBridge.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#include <algorithm>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>

#import <QuartzCore/CAMetalDisplayLink.h>

// Display callbacks run on the render thread's run loop. Keep only the latest
// update so a slow frame never accumulates a queue of obsolete drawables.
@interface RecoilDisplayLinkDelegate : NSObject <CAMetalDisplayLinkDelegate>
@property(nonatomic, strong) CAMetalDisplayLinkUpdate* pendingUpdate;
@property(nonatomic) BOOL frameInProgress;
@property(nonatomic) BOOL waitingForFrame;
@end

@implementation RecoilDisplayLinkDelegate
- (void)metalDisplayLink:(CAMetalDisplayLink*)link needsUpdate:(CAMetalDisplayLinkUpdate*)update
{
	// SDL also pumps the native run loop while polling input. Drop callbacks
	// during that frame so extra pre-acquired drawables cannot exhaust the pool
	// and block input polling before the current drawable has been presented.
	if (self.frameInProgress)
		return;
	self.pendingUpdate = update;
	// A display-link callback alone need not make NSRunLoop::runMode return.
	// Wake only our explicit frame wait, never SDL's nested event polling.
	if (self.waitingForFrame)
		CFRunLoopStop(CFRunLoopGetCurrent());
}
@end

namespace MacMetalFX {
namespace {
	id<MTLDevice> device;
	id<MTLCommandQueue> queue;
	id<MTLFXSpatialScaler> scaler;
	id<MTLTexture> inputTexture;
	id<MTLTexture> outputTexture;
	id<MTLBuffer> readback;
	NSString* errorMessage;
	CAMetalDisplayLink* displayLink;
	RecoilDisplayLinkDelegate* displayDelegate;
	CAMetalLayer* displayLayer;
	id<CAMetalDrawable> frameDrawable;
	int displayRefreshRate = 0;
	id<MTLDevice> sharedDevice;
	id<MTLCommandQueue> sharedQueue;
	id<MTLFXSpatialScaler> sharedScaler;
	id<MTLTexture> sharedOutput;
	id<MTLRenderPipelineState> presentPipeline;
	MTLPixelFormat presentFormat = MTLPixelFormatInvalid;
	id<MTLSharedEvent> mesaReady;
	id<MTLSharedEvent> metalDone;
	id<MTLCommandBuffer> pendingShared;
	uint64_t sharedEventValue = 0;
	constexpr unsigned int NUM_PRESENT_TEXTURES = 3;
	id<MTLTexture> presentTextures[NUM_PRESENT_TEXTURES];
	id<MTLCommandBuffer> presentCommands[NUM_PRESENT_TEXTURES];
	unsigned int presentIndex = 0;
	__weak CAMetalLayer* hdrLayer;
	MTLPixelFormat sdrFormat = MTLPixelFormatBGRA8Unorm;
	CGColorSpaceRef sdrColorSpace = nullptr;

	bool PrepareSharedDevice(void* handle)
	{
		id<MTLDevice> next = (__bridge id<MTLDevice>)handle;
		if (!next)
			return false;
		if (next != sharedDevice) {
			WaitShared();
			sharedScaler = nil;
			sharedOutput = nil;
			presentPipeline = nil;
			sharedDevice = next;
			sharedQueue = [next newCommandQueue];
			mesaReady = [next newSharedEvent];
			metalDone = [next newSharedEvent];
			sharedEventValue = 0;
			for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
				presentTextures[i] = nil;
				presentCommands[i] = nil;
			}
			presentIndex = 0;
		}
		if (pendingShared.status == MTLCommandBufferStatusError) {
			errorMessage = pendingShared.error.localizedDescription ?: @"shared GPU command failed";
			return false;
		}
		return sharedQueue != nil;
	}

	id<MTLTexture> Texture2D(void* handle)
	{
		id<MTLTexture> texture = (__bridge id<MTLTexture>)handle;
		if (!texture || texture.device != sharedDevice || texture.sampleCount != 1)
			return nil;
		if (texture.textureType == MTLTextureType2D)
			return texture;
		// KosmicKrisp uses one-layer arrays for input attachments.
		if (texture.textureType == MTLTextureType2DArray && texture.arrayLength == 1)
			return [texture newTextureViewWithPixelFormat:texture.pixelFormat textureType:MTLTextureType2D
				levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
		return nil;
	}

	id<MTLCommandBuffer> BeginShared(void* nativeQueue)
	{
		id<MTLCommandBuffer> commands = [sharedQueue commandBuffer];
		if (commands && nativeQueue) {
			if (@available(macOS 26.0, *)) {
				if (!mesaReady || !metalDone) {
					errorMessage = @"could not allocate shared GPU events";
					return nil;
				}
				id<MTL4CommandQueue> mesa = (__bridge id<MTL4CommandQueue>)nativeQueue;
				[mesa signalEvent:mesaReady value:++sharedEventValue];
				[commands encodeWaitForEvent:mesaReady value:sharedEventValue];
			}
		}
		return commands;
	}

	bool Complete(id<MTLCommandBuffer> commands, void* nativeQueue, bool asynchronous = false)
	{
		if (nativeQueue) {
			if (@available(macOS 26.0, *)) {
				[commands encodeSignalEvent:metalDone value:sharedEventValue];
				[commands commit];
				id<MTL4CommandQueue> mesa = (__bridge id<MTL4CommandQueue>)nativeQueue;
				// The next GL submission waits on the GPU before reusing its textures.
				[mesa waitForEvent:metalDone value:sharedEventValue];
				pendingShared = commands;
				return true;
			}
		}
		[commands commit];
		if (asynchronous) {
			pendingShared = commands;
			return true;
		}
		[commands waitUntilCompleted];
		if (commands.status != MTLCommandBufferStatusCompleted) {
			errorMessage = commands.error.localizedDescription ?: @"shared GPU command failed";
			return false;
		}
		return true;
	}
}

void StopDisplayLink()
{
	[displayLink invalidate];
	displayLink.delegate = nil;
	displayLink = nil;
	displayDelegate.pendingUpdate = nil;
	displayDelegate = nil;
	frameDrawable = nil;
	displayLayer = nil;
	displayRefreshRate = 0;
}

bool WaitForFrame(void* layerHandle, int refreshRate, bool visible)
{
	@autoreleasepool {
		CAMetalLayer* layer = (__bridge CAMetalLayer*)layerHandle;
		if (!layer || ![NSThread isMainThread])
			return false;
		if (layer != displayLayer) {
			StopDisplayLink();
			displayDelegate = [RecoilDisplayLinkDelegate new];
			displayLink = [[CAMetalDisplayLink alloc] initWithMetalLayer:layer];
			if (!displayLink)
				return false;
			displayLayer = layer;
			displayLink.delegate = displayDelegate;
			displayLink.preferredFrameLatency = 2;
			[displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
		}
		if (refreshRate != displayRefreshRate) {
			// Request the display rate explicitly. A lower minimum lets ProMotion
			// choose 60/30 Hz even when this renderer can sustain a higher rate.
			displayLink.preferredFrameRateRange = refreshRate > 0
				? CAFrameRateRangeMake(refreshRate, refreshRate, refreshRate) : CAFrameRateRangeDefault;
			displayRefreshRate = refreshRate;
		}
		frameDrawable = nil;
		displayDelegate.frameInProgress = NO;
		displayLink.paused = !visible;
		if (!visible) {
			displayDelegate.pendingUpdate = nil;
			return false;
		}
		// Occluded and minimized windows may receive no updates. Bound the wait
		// so SDL can still pump quit, focus and window-restoration events.
		NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:0.05];
		displayDelegate.waitingForFrame = YES;
		while (!displayDelegate.pendingUpdate && deadline.timeIntervalSinceNow > 0)
			[[NSRunLoop mainRunLoop] runMode:NSDefaultRunLoopMode beforeDate:deadline];
		displayDelegate.waitingForFrame = NO;
		frameDrawable = displayDelegate.pendingUpdate.drawable;
		displayDelegate.pendingUpdate = nil;
		displayDelegate.frameInProgress = frameDrawable != nil;
		return frameDrawable != nil;
	}
}

void WaitShared()
{
	[pendingShared waitUntilCompleted];
	pendingShared = nil;
}

const char* GetError()
{
	return errorMessage ? errorMessage.UTF8String : "MetalFX resource allocation failed";
}

void Destroy()
{
	StopDisplayLink();
	WaitShared();
	scaler = nil;
	inputTexture = nil;
	outputTexture = nil;
	readback = nil;
	queue = nil;
	device = nil;
	errorMessage = nil;
	sharedScaler = nil;
	sharedOutput = nil;
	presentPipeline = nil;
	sharedQueue = nil;
	sharedDevice = nil;
	presentFormat = MTLPixelFormatInvalid;
	mesaReady = nil;
	metalDone = nil;
	sharedEventValue = 0;
	for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
		presentTextures[i] = nil;
		presentCommands[i] = nil;
	}
	presentIndex = 0;
	if (hdrLayer) ConfigureHDR((__bridge void*)hdrLayer, nullptr, false);
	if (sdrColorSpace) CGColorSpaceRelease(sdrColorSpace);
	sdrColorSpace = nullptr;
}

bool UpscaleShared(void* deviceHandle, void* inputHandle, void* outputHandle, void* nativeQueue)
{
	@autoreleasepool {
		if (!PrepareSharedDevice(deviceHandle))
			return false;
		id<MTLTexture> source = Texture2D(inputHandle);
		id<MTLTexture> destination = Texture2D(outputHandle);
		if (!source || !destination ||
			destination.width < source.width || destination.height < source.height ||
			![MTLFXSpatialScalerDescriptor supportsDevice:sharedDevice]) {
			errorMessage = @"Mesa textures are incompatible with MetalFX spatial scaling";
			return false;
		}
		if (!sharedScaler || sharedScaler.inputWidth != source.width || sharedScaler.inputHeight != source.height ||
			sharedScaler.outputWidth != destination.width || sharedScaler.outputHeight != destination.height ||
			sharedScaler.colorTextureFormat != source.pixelFormat || sharedScaler.outputTextureFormat != destination.pixelFormat) {
			MTLFXSpatialScalerDescriptor* descriptor = [MTLFXSpatialScalerDescriptor new];
			descriptor.inputWidth = source.width;
			descriptor.inputHeight = source.height;
			descriptor.outputWidth = destination.width;
			descriptor.outputHeight = destination.height;
			descriptor.colorTextureFormat = source.pixelFormat;
			descriptor.outputTextureFormat = destination.pixelFormat;
			descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
			sharedScaler = [descriptor newSpatialScalerWithDevice:sharedDevice];
			sharedOutput = nil;
		}
		if (!sharedScaler)
			return false;
		id<MTLTexture> scalerOutput = destination;
		if (destination.storageMode != MTLStorageModePrivate) {
			// MetalFX requires private output; KosmicKrisp currently allocates shared heaps.
			// Copy on the GPU into GL's destination after scaling, without CPU staging.
			if (!sharedOutput) {
				MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
					destination.pixelFormat width:destination.width height:destination.height mipmapped:NO];
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.usage = sharedScaler.outputTextureUsage;
				sharedOutput = [sharedDevice newTextureWithDescriptor:descriptor];
			}
			scalerOutput = sharedOutput;
		}
		if (!scalerOutput || (source.usage & sharedScaler.colorTextureUsage) != sharedScaler.colorTextureUsage ||
			(scalerOutput.usage & sharedScaler.outputTextureUsage) != sharedScaler.outputTextureUsage) {
			errorMessage = @"Mesa textures do not provide the usage required by MetalFX";
			return false;
		}
		id<MTLCommandBuffer> commands = BeginShared(nativeQueue);
		if (!commands)
			return false;
		sharedScaler.colorTexture = source;
		sharedScaler.outputTexture = scalerOutput;
		sharedScaler.inputContentWidth = source.width;
		sharedScaler.inputContentHeight = source.height;
		[sharedScaler encodeToCommandBuffer:commands];
		if (scalerOutput != destination) {
			id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
			if (!blit)
				return false;
			[blit copyFromTexture:scalerOutput sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
				sourceSize:MTLSizeMake(destination.width, destination.height, 1) toTexture:destination
				destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
			[blit endEncoding];
		}
		const bool success = Complete(commands, nativeQueue);
		// Do not retain borrowed GL allocations across resize or context destruction.
		sharedScaler.colorTexture = nil;
		sharedScaler.outputTexture = nil;
		return success;
	}
}

float HDRHeadroom(void* nativeWindow)
{
	NSWindow* window = (__bridge NSWindow*)nativeWindow;
	return std::max(1.0f, float(window.screen.maximumExtendedDynamicRangeColorComponentValue));
}

bool ConfigureHDR(void* layerHandle, void* nativeWindow, bool enabled)
{
	CAMetalLayer* layer = (__bridge CAMetalLayer*)layerHandle;
	NSWindow* window = (__bridge NSWindow*)nativeWindow;
	if (!layer) return false;
	// Potential headroom avoids disabling HDR during the display's EDR ramp-up.
	const bool available = window.screen.maximumPotentialExtendedDynamicRangeColorComponentValue > 1.0;
	if (enabled && available) {
		if (hdrLayer != layer) {
			WaitShared();
			StopDisplayLink();
			if (hdrLayer) ConfigureHDR((__bridge void*)hdrLayer, nullptr, false);
			sdrFormat = layer.pixelFormat;
			if (sdrColorSpace) CGColorSpaceRelease(sdrColorSpace);
			sdrColorSpace = layer.colorspace ? CGColorSpaceRetain(layer.colorspace) : nullptr;
			hdrLayer = layer;
			CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
			layer.pixelFormat = MTLPixelFormatRGBA16Float;
			layer.colorspace = space;
			CGColorSpaceRelease(space);
			layer.wantsExtendedDynamicRangeContent = YES;
		}
		return true;
	}
	if (hdrLayer == layer) {
		WaitShared();
		StopDisplayLink();
		layer.wantsExtendedDynamicRangeContent = NO;
		layer.pixelFormat = sdrFormat;
		layer.colorspace = sdrColorSpace;
		hdrLayer = nil;
	}
	return false;
}

static bool PresentTexture(void* deviceHandle, void* inputHandle, void* layerHandle, float headroom, void* nativeQueue, bool flipY, bool asynchronous)
{
	@autoreleasepool {
		if (!PrepareSharedDevice(deviceHandle))
			return false;
		id<MTLTexture> source = Texture2D(inputHandle);
		CAMetalLayer* layer = (__bridge CAMetalLayer*)layerHandle;
		if (!source || !layer || layer.device != sharedDevice || !(source.usage & MTLTextureUsageShaderRead)) {
			errorMessage = @"Mesa texture and SDL Metal layer cannot share a device";
			return false;
		}
		if (!presentPipeline || presentFormat != layer.pixelFormat) {
			NSString* shader = @R"(
#include <metal_stdlib>
using namespace metal;
struct Vertex { float4 position [[position]]; float2 uv; };
vertex Vertex presentVertex(uint index [[vertex_id]]) {
    const float2 points[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    Vertex result;
    result.position = float4(points[index], 0, 1);
    result.uv = float2(points[index].x * 0.5 + 0.5, 0.5 - points[index].y * 0.5);
    return result;
}
fragment float4 presentFragment(Vertex interpolated [[stage_in]], texture2d<float> color [[texture(0)]],
    constant float& headroom [[buffer(0)]], constant uint& flipY [[buffer(1)]]) {
    constexpr sampler sampleColor(filter::linear, address::clamp_to_edge);
    float2 uv = interpolated.uv;
    // Zink's pbuffer is presentation-oriented; an offscreen GL framebuffer is
    // bottom-first. HDR presents that offscreen composition target directly.
    if (headroom > 0.0 || flipY) uv.y = 1.0 - uv.y;
    float3 rgb = max(color.sample(sampleColor, uv).rgb, 0.0);
    if (headroom > 0.0) {
        // The game and legacy UI blend in display-referred sRGB. Decode once,
        // after compositing. UI white remains SDR white (linear 1.0).
        rgb = select(pow((rgb + 0.055) / 1.055, float3(2.4)), rgb / 12.92, rgb <= 0.04045);
        float peak = max(max(rgb.r, rgb.g), rgb.b);
        if (peak > 1.0) {
            float range = max(headroom - 1.0, 0.0001);
            float mapped = 1.0 + range * (1.0 - exp(-(peak - 1.0) / range));
            rgb *= min(mapped, headroom) / peak;
        }
    }
    return float4(rgb, 1);
}
)";
			NSError* error = nil;
			id<MTLLibrary> library = [sharedDevice newLibraryWithSource:shader options:nil error:&error];
			MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
			descriptor.vertexFunction = [library newFunctionWithName:@"presentVertex"];
			descriptor.fragmentFunction = [library newFunctionWithName:@"presentFragment"];
			descriptor.colorAttachments[0].pixelFormat = layer.pixelFormat;
			if (library)
				presentPipeline = [sharedDevice newRenderPipelineStateWithDescriptor:descriptor error:&error];
			if (!presentPipeline) {
				errorMessage = error.localizedDescription ?: @"could not create shared presentation pipeline";
				return false;
			}
			presentFormat = layer.pixelFormat;
		}
		const bool displayLinked = displayLink && layer == displayLayer;
		id<CAMetalDrawable> drawable = displayLinked ? frameDrawable : [layer nextDrawable];
		if (displayLinked) frameDrawable = nil;
		if (!drawable)
			return true; // Minimized or occluded windows may not have a drawable.
		id<MTLCommandBuffer> commands = BeginShared(nativeQueue);
		if (!commands)
			return false;
		MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
		pass.colorAttachments[0].texture = drawable.texture;
		pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
		if (!encoder)
			return false;
		[encoder setRenderPipelineState:presentPipeline];
		[encoder setFragmentTexture:source atIndex:0];
		[encoder setFragmentBytes:&headroom length:sizeof(headroom) atIndex:0];
		const uint32_t flip = flipY;
		[encoder setFragmentBytes:&flip length:sizeof(flip) atIndex:1];
		[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[encoder endEncoding];
		if (!displayLinked) [commands presentDrawable:drawable];
		// GL may overwrite the same pbuffer on the next frame.
		const bool success = Complete(commands, nativeQueue, asynchronous);
		if (success && displayLinked) [drawable present];
		if (displayLinked) displayDelegate.frameInProgress = NO;
		return success;
	}
}

bool PresentShared(void* deviceHandle, void* inputHandle, void* layerHandle, float headroom, void* nativeQueue)
{
	return PresentTexture(deviceHandle, inputHandle, layerHandle, headroom, nativeQueue, false, false);
}

bool PresentPixels(const void* rgba, int width, int height, void* layerHandle)
{
	@autoreleasepool {
		CAMetalLayer* layer = (__bridge CAMetalLayer*)layerHandle;
		if (!rgba || width <= 0 || height <= 0 || !layer ||
			!PrepareSharedDevice((__bridge void*)layer.device))
			return false;
		const unsigned int slot = presentIndex % NUM_PRESENT_TEXTURES;
		// Reuse only a completed upload. Normally drawable throttling has already
		// retired this slot, so no new frame-sized allocation or GPU blit is needed.
		[presentCommands[slot] waitUntilCompleted];
		if (presentCommands[slot].status == MTLCommandBufferStatusError) {
			errorMessage = presentCommands[slot].error.localizedDescription ?: @"pixel presentation command failed";
			return false;
		}
		presentCommands[slot] = nil;
		id<MTLTexture> texture = presentTextures[slot];
		if (!texture || texture.width != width || texture.height != height) {
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
				MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
			descriptor.storageMode = MTLStorageModeShared;
			descriptor.usage = MTLTextureUsageShaderRead;
			texture = [sharedDevice newTextureWithDescriptor:descriptor];
			presentTextures[slot] = texture;
		}
		if (!texture) {
			errorMessage = @"could not allocate reusable presentation texture";
			return false;
		}
		[texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0
			withBytes:rgba bytesPerRow:size_t(width) * 4];
		if (!PresentTexture((__bridge void*)sharedDevice, (__bridge void*)texture, layerHandle, 0.0f, nullptr, true, true))
			return false;
		presentCommands[slot] = pendingShared;
		++presentIndex;
		return true;
	}
}

bool Upscale(const void* rgba, int width, int height, int outputWidth, int outputHeight, const void*& output, int& outputRowLength)
{
	output = nullptr;
	if (!rgba || width <= 0 || height <= 0 || outputWidth < width || outputHeight < height)
		return false;

	@autoreleasepool {
		if (!device) {
			device = MTLCreateSystemDefaultDevice();
			if (!device || ![MTLFXSpatialScalerDescriptor supportsDevice:device]) {
				errorMessage = @"spatial upscaling is unsupported on this GPU";
				return false;
			}
			queue = [device newCommandQueue];
		}
		if (!queue)
			return false;

		if (!scaler || scaler.inputWidth != width || scaler.inputHeight != height ||
			scaler.outputWidth != outputWidth || scaler.outputHeight != outputHeight) {
			MTLFXSpatialScalerDescriptor* descriptor = [MTLFXSpatialScalerDescriptor new];
			descriptor.inputWidth = width;
			descriptor.inputHeight = height;
			descriptor.outputWidth = outputWidth;
			descriptor.outputHeight = outputHeight;
			descriptor.colorTextureFormat = MTLPixelFormatRGBA8Unorm;
			descriptor.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
			// The legacy scene is already display-referred; don't apply gamma again.
			descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
			scaler = [descriptor newSpatialScalerWithDevice:device];
			if (!scaler)
				return false;

			MTLTextureDescriptor* texture = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
				MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
			texture.storageMode = MTLStorageModeShared;
			texture.usage = scaler.colorTextureUsage;
			inputTexture = [device newTextureWithDescriptor:texture];
			texture.width = outputWidth;
			texture.height = outputHeight;
			texture.storageMode = MTLStorageModePrivate;
			texture.usage = scaler.outputTextureUsage;
			outputTexture = [device newTextureWithDescriptor:texture];
			const NSUInteger rowBytes = (NSUInteger(outputWidth) * 4 + 255) & ~NSUInteger(255);
			readback = [device newBufferWithLength:rowBytes * outputHeight options:MTLResourceStorageModeShared];
		}
		if (!inputTexture || !outputTexture || !readback)
			return false;

		[inputTexture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:rgba bytesPerRow:NSUInteger(width) * 4];
		scaler.colorTexture = inputTexture;
		scaler.outputTexture = outputTexture;
		scaler.inputContentWidth = width;
		scaler.inputContentHeight = height;
		id<MTLCommandBuffer> commands = [queue commandBuffer];
		if (!commands)
			return false;
		[scaler encodeToCommandBuffer:commands];
		id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
		if (!blit)
			return false;
		const NSUInteger rowBytes = (NSUInteger(outputWidth) * 4 + 255) & ~NSUInteger(255);
		[blit copyFromTexture:outputTexture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			sourceSize:MTLSizeMake(outputWidth, outputHeight, 1) toBuffer:readback destinationOffset:0
			destinationBytesPerRow:rowBytes destinationBytesPerImage:rowBytes * outputHeight];
		[blit endEncoding];
		[commands commit];
		// UI continues in Mesa GL, so the upscaled world must return before its draw.
		[commands waitUntilCompleted];
		if (commands.status == MTLCommandBufferStatusError) {
			errorMessage = commands.error.localizedDescription;
			return false;
		}
		output = readback.contents;
		outputRowLength = rowBytes / 4;
		return true;
	}
}
}
