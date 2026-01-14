#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_raii.hpp>
#include <random>

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_vulkan.h"

#include "volkan.h"
#include "vk_utils.h"

#define KHRONOS_VALIDATION_LAYER "VK_LAYER_KHRONOS_validation"

#ifdef NDEBUG
const bool DEBUG = false;
#else
const bool DEBUG = true;
#endif


// CREATE INSTANCE UTILS
PFN_vkCreateDebugUtilsMessengerEXT pfnVkCreateDebugUtilsMessengerEXT;
PFN_vkDestroyDebugUtilsMessengerEXT pfnVkDestroyDebugUtilsMessengerEXT;

#ifndef VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger) {
      return pfnVkCreateDebugUtilsMessengerEXT(instance, pCreateInfo,
                                               pAllocator, pMessenger);
    }
#endif

    VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(
        VkInstance instance, VkDebugUtilsMessengerEXT messenger,
        VkAllocationCallbacks const* pAllocator) {
      return pfnVkDestroyDebugUtilsMessengerEXT(instance, messenger,
                                                pAllocator);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  vk::DebugUtilsMessageTypeFlagsEXT messageType,
                  const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                  void* pUserData) {
      std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

      return VK_FALSE;
    }


vk::DebugUtilsMessengerCreateInfoEXT debugUtils() {
  vk::DebugUtilsMessengerCreateInfoEXT createInfo;
  createInfo.messageSeverity =
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
  createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
  createInfo.pfnUserCallback = debugCallback;
  return createInfo;
}

std::vector<const char*> getRequiredExtensions() {
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char*> extensions(glfwExtensions,
                                      glfwExtensions + glfwExtensionCount);

  if (DEBUG) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

void Volkan::init(GLFWwindow* window) {
  window_ = window;
  createInstance();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createCommandPool();
  createDepthResources();
  createFramebuffers();
  createDescriptorPool();
  createFrameData();
}

void Volkan::createInstance() {
  vk::ApplicationInfo appInfo("vkEngine", VK_MAKE_API_VERSION(0, 1, 3, 0),
                              "No Engine", VK_MAKE_API_VERSION(0, 1, 0, 0),
                              VK_API_VERSION_1_3);
  auto extensions = getRequiredExtensions();

#ifdef NDEBUG
  vk::StructureChain<vk::InstanceCreateInfo> instanceCreateInfoChain(
      {{}, &appInfo, {}, extensions});
#else
  if (!checkValidationLayerSupport(KHRONOS_VALIDATION_LAYER)) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  const std::vector<const char*> layer_names = {KHRONOS_VALIDATION_LAYER};
  vk::InstanceCreateInfo instanceCI;
  instanceCI.setPApplicationInfo(&appInfo);
  instanceCI.setPEnabledLayerNames(layer_names);
  instanceCI.setPEnabledExtensionNames(extensions);

  vk::StructureChain<vk::InstanceCreateInfo,
                     vk::DebugUtilsMessengerCreateInfoEXT>
      instanceCreateInfoChain(instanceCI, debugUtils());
#endif

  vk::raii::Context context;
  instance_ = context.createInstance(
      instanceCreateInfoChain.get<vk::InstanceCreateInfo>());
  debugMessenger_ = instance_.createDebugUtilsMessengerEXT(debugUtils());
}

void Volkan::createSurface() {
  VkSurfaceKHR _surface;
  glfwCreateWindowSurface(*instance_, window_, nullptr, &_surface);
  surface_ = vk::raii::SurfaceKHR(instance_, _surface);
}

void Volkan::pickPhysicalDevice() {
  vk::raii::PhysicalDevices physicalDevices(instance_);
  for (vk::raii::PhysicalDevice device : physicalDevices) {
    if (isDeviceSuitable(device, surface_, extensions_)) {
      physicalDevice_ = device;
      return;
    }
  }

  throw std::runtime_error("failed to find a suitable GPU!");
}

void Volkan::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_, surface_);
  std::set<uint32_t> uniqueQueueFamilies = {
      indices.graphicsFamily.value(), indices.presentFamily.value()};

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCI;
    queueCI.setQueueFamilyIndex(queueFamily);
    queueCI.setQueuePriorities(queuePriority);
    queueCreateInfos.push_back(queueCI);
  }

  vk::PhysicalDeviceFeatures deviceFeatures{};
  vk::PhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures;
  meshShaderFeatures.setTaskShader(true);
  meshShaderFeatures.setMeshShader(true);

  vk::DeviceCreateInfo deviceCI;
  deviceCI.setQueueCreateInfos(queueCreateInfos);
  deviceCI.setPEnabledFeatures(&deviceFeatures);
  deviceCI.setPNext(&meshShaderFeatures);
  deviceCI.setPEnabledExtensionNames(extensions_);
#ifndef NDEBUG
  const std::vector<const char*>layer_names = {KHRONOS_VALIDATION_LAYER};
  deviceCI.setPEnabledLayerNames(layer_names);
#endif

  device_ = physicalDevice_.createDevice(deviceCI);

  graphicsQueue_ = device_.getQueue(indices.graphicsFamily.value(), 0);
  presentQueue_ = device_.getQueue(indices.presentFamily.value(), 0);
}

void Volkan::createSwapChain() {
  SwapChainSupportDetails swapChainSupport =
      querySwapChainSupport(physicalDevice_, surface_);

  vk::SurfaceFormatKHR surfaceFormat =
      chooseSwapSurfaceFormat(swapChainSupport.formats);
  vk::PresentModeKHR presentMode =
      chooseSwapPresentMode(swapChainSupport.presentModes);
  vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window_);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_, surface_);
  uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                    indices.presentFamily.value()};

  vk::SharingMode imageSharingMode;
  if (indices.graphicsFamily != indices.presentFamily) {
    imageSharingMode = vk::SharingMode::eConcurrent;
  } else {
    imageSharingMode = vk::SharingMode::eExclusive;
  }

  vk::SwapchainCreateInfoKHR swapchainCI;
  swapchainCI.setSurface(*surface_);
  swapchainCI.setMinImageCount(imageCount);
  swapchainCI.setImageFormat(surfaceFormat.format);
  swapchainCI.setImageColorSpace(surfaceFormat.colorSpace);
  swapchainCI.setImageExtent(extent);
  swapchainCI.setImageArrayLayers(1);
  swapchainCI.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
  swapchainCI.setImageSharingMode(imageSharingMode);
  swapchainCI.setPQueueFamilyIndices(queueFamilyIndices);
  swapchainCI.setQueueFamilyIndexCount(2);
  swapchainCI.setPreTransform(swapChainSupport.capabilities.currentTransform);
  swapchainCI.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
  swapchainCI.setPresentMode(presentMode);
  swapchainCI.setClipped(true);

  swapChain_ = device_.createSwapchainKHR(swapchainCI);
  swapChainImages_ = swapChain_.getImages();
  swapChainImageFormat_ = surfaceFormat.format;
  swapChainExtent_ = extent;
}

void Volkan::createImageViews() {
  swapChainImageViews_.clear();
  for (vk::Image image : swapChainImages_) {
    vk::ImageViewCreateInfo imageViewCI;
    imageViewCI.setImage(image);
    imageViewCI.setViewType(vk::ImageViewType::e2D);
    imageViewCI.setFormat(swapChainImageFormat_);
    imageViewCI.setComponents(
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity});
    imageViewCI.setSubresourceRange(
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    swapChainImageViews_.push_back(device_.createImageView(imageViewCI));
  }
}

void Volkan::createRenderPass() {
  vk::AttachmentDescription colorAttachment(
      {}, swapChainImageFormat_, vk::SampleCountFlagBits::e1,
      vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
      vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
      vk::ImageLayout::ePresentSrcKHR);
  
  vk::AttachmentReference colorAttachmentRef(
      0, vk::ImageLayout::eColorAttachmentOptimal);
  
  vk::AttachmentDescription depthAttachment(
      {}, findDepthFormat(physicalDevice_), vk::SampleCountFlagBits::e1,
      vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
      vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eDepthStencilAttachmentOptimal);
  
  vk::AttachmentReference depthAttachmentRef(
      1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  
  vk::SubpassDescription subpass;
  subpass.setColorAttachments(colorAttachmentRef);
  subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
  subpass.setPDepthStencilAttachment(&depthAttachmentRef);
  
  vk::SubpassDependency subpassDependency(
      vk::SubpassExternal, 0,
      vk::PipelineStageFlagBits::eColorAttachmentOutput |
      vk::PipelineStageFlagBits::eLateFragmentTests,
      vk::PipelineStageFlagBits::eColorAttachmentOutput |
      vk::PipelineStageFlagBits::eEarlyFragmentTests,
      vk::AccessFlagBits::eNone |
      vk::AccessFlagBits::eDepthStencilAttachmentWrite,
      vk::AccessFlagBits::eColorAttachmentWrite |
      vk::AccessFlagBits::eDepthStencilAttachmentWrite,
      {});
  
  std::array<vk::AttachmentDescription, 2> attachments =
  {colorAttachment,depthAttachment}; vk::RenderPassCreateInfo
  renderPassCreateInfo;
  renderPassCreateInfo.setAttachments(attachments);
  renderPassCreateInfo.setSubpasses(subpass);
  renderPassCreateInfo.setDependencies(subpassDependency);
  
  renderPass_ = device_.createRenderPass(renderPassCreateInfo);
}

vk::raii::DescriptorSetLayout Volkan::createDescriptorSetLayout(vk::DescriptorSetLayoutBinding layoutBinding) {
  return device_.createDescriptorSetLayout({{}, layoutBinding});
}

vk::raii::DescriptorSets Volkan::createDescriptorSets(
    vk::raii::DescriptorSetLayout& layout) {
  const std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, layout);
  vk::DescriptorSetAllocateInfo allocInfo;
  allocInfo.setDescriptorPool(*descriptorPool_);
  allocInfo.setSetLayouts(layouts);
  vk::raii::DescriptorSets dsets(device_, allocInfo);
  return dsets;
}

BufferData Volkan::allocateUniformBuffer(vk::raii::DescriptorSet& dset, uint64_t size) {
  vk::BufferCreateInfo bufferCI;
  bufferCI.setSize(size);
  bufferCI.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
  bufferCI.setSharingMode(vk::SharingMode::eExclusive);
  vk::raii::Buffer buffer(device_, bufferCI);

  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

  uint32_t memoryType =
      findMemoryType(memRequirements.memoryTypeBits,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent, physicalDevice_);

  vk::raii::DeviceMemory memory(device_, {memRequirements.size, memoryType});

  buffer.bindMemory(memory, 0);

  void* mapped = memory.mapMemory(0, size);

  vk::DescriptorBufferInfo info(*buffer, 0, size);
  vk::WriteDescriptorSet descriptorWrite;
  descriptorWrite.setBufferInfo(info);
  descriptorWrite.setDstSet(*dset);
  descriptorWrite.setDstBinding(0);
  descriptorWrite.setDstArrayElement(0);
  descriptorWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
  device_.updateDescriptorSets(descriptorWrite, {});
  
  return BufferData(buffer, memory, mapped);
}

void Volkan::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_, surface_);
  commandPool_ = device_.createCommandPool(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       queueFamilyIndices.graphicsFamily.value()});
}

void Volkan::createDepthResources() {
  vk::Format format = findDepthFormat(physicalDevice_);

  vk::ImageCreateInfo imageCreateInfo{
      {},
      vk::ImageType::e2D,
      format,
      {swapChainExtent_.width, swapChainExtent_.height, 1},
      1,
      1,
      vk::SampleCountFlagBits::e1,
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment,
      vk::SharingMode::eExclusive};
  depthImage_ = device_.createImage(imageCreateInfo);

  vk::MemoryRequirements memRequirements = depthImage_.getMemoryRequirements();

  depthImageMemory_ = device_.allocateMemory(
      {memRequirements.size,
       findMemoryType(memRequirements.memoryTypeBits,
                      vk::MemoryPropertyFlagBits::eDeviceLocal,
                      physicalDevice_)});

  vk::BindImageMemoryInfo bindInfo;
  bindInfo.setImage(*depthImage_);
  bindInfo.setMemory(*depthImageMemory_);
  bindInfo.setMemoryOffset(0);
  device_.bindImageMemory2(bindInfo);

  vk::ImageViewCreateInfo imageViewCI;
  imageViewCI.setImage(*depthImage_);
  imageViewCI.setViewType(vk::ImageViewType::e2D);
  imageViewCI.setFormat(format);
  imageViewCI.setComponents(
      {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
       vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity});
  imageViewCI.setSubresourceRange(
      {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});

  depthImageView_ = device_.createImageView(imageViewCI);
}

void Volkan::createFramebuffers() {
  swapChainFramebuffers_.clear();
  for (size_t i = 0; i < swapChainImageViews_.size(); i++) {
    std::array<vk::ImageView, 2> attachments = {swapChainImageViews_[i],
                                                depthImageView_};
    vk::FramebufferCreateInfo framebufferInfo;
    framebufferInfo.setAttachments(attachments);
    framebufferInfo.setRenderPass(*renderPass_);
    framebufferInfo.setWidth(swapChainExtent_.width);
    framebufferInfo.setHeight(swapChainExtent_.height);
    framebufferInfo.setLayers(1);
    swapChainFramebuffers_.push_back(device_.createFramebuffer(framebufferInfo));
  }
}

void Volkan::createDescriptorPool() {
  // Lots of pool size for imgui
  vk::DescriptorPoolSize poolsize[] = {
      //{vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT},
      {vk::DescriptorType::eSampler, 1000},
      {vk::DescriptorType::eCombinedImageSampler, 1000},
      {vk::DescriptorType::eSampledImage, 1000},
      {vk::DescriptorType::eStorageImage, 1000},
      {vk::DescriptorType::eUniformTexelBuffer, 1000},
      {vk::DescriptorType::eStorageTexelBuffer, 1000},
      {vk::DescriptorType::eUniformBuffer, 1000},
      {vk::DescriptorType::eStorageBuffer, 1000},
      {vk::DescriptorType::eUniformBufferDynamic, 1000},
      {vk::DescriptorType::eStorageBufferDynamic, 1000},
      {vk::DescriptorType::eInputAttachment, 1000},
  };
  descriptorPool_ = device_.createDescriptorPool(
      {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1000, poolsize});
}

BufferData Volkan::createBuffer(vk::DeviceSize bufferSize, vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags properties) {
  vk::raii::Buffer buffer(device_,
                          {{}, bufferSize, usage, vk::SharingMode::eExclusive});

  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

  uint32_t memoryType =
      findMemoryType(memRequirements.memoryTypeBits,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent, physicalDevice_);

  vk::raii::DeviceMemory memory(device_, {memRequirements.size, memoryType});

  buffer.bindMemory(memory, 0);

  return BufferData(buffer, memory, (void*)nullptr);
}

BufferData Volkan::createUniformBuffer(uint64_t size) {
  BufferData bufferData =
      createBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer,
                   vk::MemoryPropertyFlagBits::eHostCoherent |
                       vk::MemoryPropertyFlagBits::eHostVisible);
  bufferData.mapped = bufferData.memory.mapMemory(0, size);
  return bufferData;
}

void Volkan::createFrameData() {
  vk::raii::CommandBuffers commandBuffers(
      device_,
      {commandPool_, vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT});

  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::raii::Semaphore imageAvailableSemaphore(device_, {{}, nullptr});
    vk::raii::Semaphore renderFinishedSemaphore(device_, {{}, nullptr});
    vk::raii::Fence inFlightFence(device_, {vk::FenceCreateFlagBits::eSignaled});
    


    FrameData frameData({std::move(commandBuffers[i]),
                         std::move(imageAvailableSemaphore),
                         std::move(renderFinishedSemaphore),
                         std::move(inFlightFence),
                         0});
    frameDatas_.push_back(std::move(frameData));
  }
}

void Volkan::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwWaitEvents();
    glfwGetFramebufferSize(window_, &width, &height);
  }

  device_.waitIdle();
  swapChain_ = nullptr;
  createSwapChain();
  createImageViews();
  createFramebuffers();
}

bool Volkan::nextFrameReady() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto _ = device_.waitForFences({fd.inFlightFence}, VK_TRUE, UINT64_MAX);

  vk::Result result;
  std::tie(result, frameDatas_[currentFrame_].imageIndex) =
      swapChain_.acquireNextImage(UINT64_MAX, fd.imageAvailableSemaphore);

  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR || framebufferResized_) {
    framebufferResized_ = false;
    recreateSwapChain();
    return false;
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }
  device_.resetFences({fd.inFlightFence});
  fd.commandBuffer.reset();
  return true;
}

void Volkan::beginCommandBuffer() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;

  commandBuffer.begin({});
}

void Volkan::beginRenderPass() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;

  vk::Rect2D renderArea({{0, 0}, swapChainExtent_});
  std::array<vk::ClearValue, 2> clearValues{};
  clearValues[0].setColor({0.0f, 0.0f, 0.0f, 1.0f});
  clearValues[1].setDepthStencil({1.0f, 0});
  vk::RenderPassBeginInfo renderPassInfo;
  renderPassInfo.setClearValues(clearValues);
  renderPassInfo.setRenderArea(renderArea);
  renderPassInfo.setRenderPass(*renderPass_);
  renderPassInfo.setFramebuffer(*swapChainFramebuffers_[fd.imageIndex]);

  commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
}

void Volkan::bindPipeline(Camera camera, glm::float32 time,
                          Material& material) {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, material.pipeline);

  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                   material.pipelineLayout, 0,
                                   *material.descriptorSets[currentFrame_], {});

  vk::Viewport viewport(0.0f, 0.0f, swapChainExtent_.width,
                        swapChainExtent_.height, 0.0f, 1.0f);
  vk::Rect2D renderArea({{0, 0}, swapChainExtent_});
  commandBuffer.setViewport(0, viewport);
  commandBuffer.setScissor(0, renderArea);

  PerFrameVariables pfv = {};
  pfv.time = time;
  pfv.view = camera.getViewMatrix();
  pfv.proj = camera.getProjectionMatrix(
      swapChainExtent_.width / (float)swapChainExtent_.height);
  pfv.cameraPos = glm::vec4(camera.pos, 1.0);

  commandBuffer.pushConstants<PerFrameVariables>(
      material.pipelineLayout,
      vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment, 0, pfv);
}

void Volkan::drawMesh() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;

  commandBuffer.drawMeshTasksEXT(1, 1, 1);
}

void Volkan::submit() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;
  vk::PipelineStageFlags waitDestinationStageMask(
      vk::PipelineStageFlagBits::eColorAttachmentOutput);
  
  commandBuffer.endRenderPass();
  commandBuffer.end();

  vk::SubmitInfo submitInfo(*fd.imageAvailableSemaphore,
                            waitDestinationStageMask, *fd.commandBuffer,
                            *fd.renderFinishedSemaphore);

  graphicsQueue_.submit(submitInfo, fd.inFlightFence);
  presentQueue_.presentKHR(
      {*fd.renderFinishedSemaphore, *swapChain_, fd.imageIndex});

  currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

vk::raii::ShaderModule Volkan::createShaderModule(const std::string& filename) {
  auto code = readFile(filename);
  return device_.createShaderModule(
      {{}, code.size(), reinterpret_cast<const uint32_t*>(code.data())});
}

vk::Viewport Volkan::getViewport()
{
  return vk::Viewport(0.0f, 0.0f, swapChainExtent_.width,
                      swapChainExtent_.height, 0.0f, 1.0f);
}
vk::Rect2D Volkan::getScissor() { return vk::Rect2D({0, 0}, swapChainExtent_); }


vk::raii::PipelineLayout Volkan::createPipelineLayout(vk::PipelineLayoutCreateInfo plci) {
  return device_.createPipelineLayout(plci);
}

vk::raii::Pipeline Volkan::createPipeline(vk::GraphicsPipelineCreateInfo gpci) {
  return device_.createGraphicsPipeline(nullptr, gpci);
}

void Volkan::cleanupRenderResources() {
    // Destroy framebuffers and render pass before device destruction
    swapChainFramebuffers_.clear();
    renderPass_ = nullptr;
}