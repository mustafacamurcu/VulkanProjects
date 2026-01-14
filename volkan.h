#pragma once
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

#include "structs.h"
#include "material.h"

struct BufferData {
  vk::raii::Buffer buffer = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
  void* mapped = nullptr;

  BufferData(vk::raii::Buffer& _buffer, vk::raii::DeviceMemory& _memory,
             void* mapped_)
      : buffer(std::move(_buffer)),
        memory(std::move(_memory)),
        mapped(mapped_) {}

  BufferData() : buffer(nullptr), memory(nullptr), mapped(nullptr) {}
};

class Volkan {
public:
  void init(GLFWwindow* window);

  vk::Viewport getViewport();
  vk::Rect2D getScissor();
  vk::raii::ShaderModule createShaderModule(const std::string& filename);
  vk::raii::DescriptorSetLayout createDescriptorSetLayout(vk::DescriptorSetLayoutBinding layoutBinding);
  vk::raii::DescriptorSets createDescriptorSets(vk::raii::DescriptorSetLayout& layout);
  vk::raii::Pipeline createPipeline(vk::GraphicsPipelineCreateInfo gpci);
  vk::raii::PipelineLayout createPipelineLayout(vk::PipelineLayoutCreateInfo plci);
  BufferData allocateUniformBuffer(vk::raii::DescriptorSet& dset, uint64_t size);

  bool nextFrameReady();

  void beginCommandBuffer();
  void beginRenderPass();
  void bindPipeline(Camera camera, glm::float32 time, Material& material);
  void drawMesh();
  void submit();

  vk::raii::Pipeline createMeshPipeline();

  
  struct FrameData {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailableSemaphore;
    vk::raii::Semaphore renderFinishedSemaphore;
    vk::raii::Fence inFlightFence;
    uint32_t imageIndex;
  };
  
  const std::vector<const char*> extensions_ = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME};
  const uint32_t WIDTH = 1920;
  const uint32_t HEIGHT = 1080;
  bool framebufferResized_ = false;

  vk::raii::RenderPass renderPass_ = nullptr;

  void cleanupRenderResources();

private:
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createSpecialization();
    void createCommandPool();
    void createDepthResources();
    void createFramebuffers();
    void createDescriptorPool();
    void createFrameData();
    void createWaves();

    BufferData createUniformBuffer(uint64_t size);
    BufferData createBuffer(vk::DeviceSize bufferSize,
                            vk::BufferUsageFlags usage,
                            vk::MemoryPropertyFlags properties);

    void recreateSwapChain();

    void updatePushConstants(FrameData& fd);

    GLFWwindow* window_;
    vk::raii::Instance instance_ = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
    vk::raii::SurfaceKHR surface_ = nullptr;
    vk::raii::PhysicalDevice physicalDevice_ = nullptr;
    vk::raii::Device device_ = nullptr;
    vk::raii::Queue graphicsQueue_ = nullptr;
    vk::raii::Queue presentQueue_ = nullptr;
    vk::raii::SwapchainKHR swapChain_ = nullptr;
    std::vector<vk::Image> swapChainImages_;
    vk::Format swapChainImageFormat_;
    vk::Extent2D swapChainExtent_;
    std::vector<vk::raii::ImageView> swapChainImageViews_;
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers_;
    vk::raii::CommandPool commandPool_ = nullptr;
    vk::raii::DescriptorPool descriptorPool_ = nullptr;
    std::vector<FrameData> frameDatas_;
    vk::raii::Image depthImage_ = nullptr;
    vk::raii::ImageView depthImageView_ = nullptr;
    vk::raii::DeviceMemory depthImageMemory_ = nullptr;


    size_t currentFrame_ = 0;
};
