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

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;
#define NUM_WAVES 15
#define OCEAN_WIDTH 1000
#define OCEAN_CELL_WIDTH 1.0f

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete() const {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

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



    class Engine {
     public:
      void run() {
        initWindow();
        initVulkan();
        initImGui();

        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        while (!glfwWindowShouldClose(window)) {
          glfwPollEvents();
          processInput(window);

          ImGui_ImplVulkan_NewFrame();
          ImGui_ImplGlfw_NewFrame();
          ImGui::NewFrame();
          ImGui::Begin("Another Window");
          ImGui::Text("Hello from another window!");
          ImGui::End();
          ImGui::Render();
          ImDrawData* draw_data = ImGui::GetDrawData();

          drawFrame(draw_data);


        }

        cleanup();
        device.waitIdle();
      }

      void drawFrame(ImDrawData* draw_data) {
        FrameData& fd = frameDatas[currentFrame];
        auto _ = device.waitForFences({fd.inFlightFence}, VK_TRUE, UINT64_MAX);

        vk::Result result;
        std::tie(result, frameDatas[currentFrame].imageIndex) =
            swapChain.acquireNextImage(UINT64_MAX, fd.imageAvailableSemaphore);

        if (result == vk::Result::eErrorOutOfDateKHR ||
            result == vk::Result::eSuboptimalKHR || framebufferResized) {
          framebufferResized = false;
          recreateSwapChain();
          return;
        } else if (result != vk::Result::eSuccess) {
          throw std::runtime_error("failed to acquire swap chain image!");
        }

        device.resetFences({fd.inFlightFence});

        fd.commandBuffer.reset();

        updatePushConstants(fd);

        recordCommandBuffer(fd.commandBuffer, fd.imageIndex, currentFrame, draw_data);

        updateWaves(fd);

        vk::PipelineStageFlags waitDestinationStageMask(
            vk::PipelineStageFlagBits::eColorAttachmentOutput);

        vk::SubmitInfo submitInfo(*fd.imageAvailableSemaphore,
                                  waitDestinationStageMask, *fd.commandBuffer,
                                  *fd.renderFinishedSemaphore);

        graphicsQueue.submit(submitInfo, fd.inFlightFence);
        presentQueue.presentKHR(
            {*fd.renderFinishedSemaphore, *swapChain, fd.imageIndex});

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
      }

      static void framebufferResizeCallback(GLFWwindow* window, int width,
                                            int height) {
        auto scene =
            reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
        scene->framebufferResized = true;
      }

      bool framebufferResized = false;

      static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
        Engine* engine =
            reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
        if (engine->firstMouse) {
          engine->lastX = xpos;
          engine->lastY = ypos;
          engine->firstMouse = false;
        }
        float xoffset = -static_cast<float>(xpos - engine->lastX);
        float yoffset = -static_cast<float>(
            engine->lastY - ypos);
        engine->lastX = xpos;
        engine->lastY = ypos;

        xoffset *= engine->mouseSensitivity;
        yoffset *= engine->mouseSensitivity;

        engine->yaw += xoffset;
        engine->pitch += yoffset;

        if (engine->pitch > 89.0f) engine->pitch = 89.0f;
        if (engine->pitch < -89.0f) engine->pitch = -89.0f;
      }

     private:
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

      struct PushConstants {
        glm::float32 time;
        alignas(16) glm::mat4 view;
        glm::mat4 proj;
        glm::vec4 cameraPos;
      };

      struct FrameData {
        vk::raii::CommandBuffer commandBuffer;
        vk::raii::Semaphore imageAvailableSemaphore;
        vk::raii::Semaphore renderFinishedSemaphore;
        vk::raii::Fence inFlightFence;
        BufferData wavesBuffer;
        PushConstants pushConstants;
        vk::raii::DescriptorSet descriptorSet;
        uint32_t imageIndex;
      };

      struct SpecializationData {
        int num_waves;
        int ocean_width;
        float ocean_cell_width;
      };

      struct alignas(32) Wave {
        float amp;
        float phase;
        float dir_x;
        float dir_z;
        float freq;
        float sharpness;
      };

      struct WavesUBO {
        Wave waves[NUM_WAVES];
      };

      struct Vertex {
        glm::vec2 pos;
        glm::vec3 color;

        static vk::VertexInputBindingDescription getBindingDescription() {
          vk::VertexInputBindingDescription bindingDescription{
              0, sizeof(Vertex), vk::VertexInputRate::eVertex};
          return bindingDescription;
        }

        static std::array<vk::VertexInputAttributeDescription, 2>
        getAttributeDescriptions() {
          std::array<vk::VertexInputAttributeDescription, 2>
              attributeDescriptions{
                  {{0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)},
                   {1, 0, vk::Format::eR32G32B32Sfloat,
                    offsetof(Vertex, color)}}};

          return attributeDescriptions;
        }
      };

      std::vector<Vertex> vertices;
      std::vector<uint32_t> indices;

      GLFWwindow* window;
      vk::raii::Instance instance = nullptr;
      vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
      vk::raii::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
      vk::raii::Device device = nullptr;
      vk::raii::SurfaceKHR surface = nullptr;
      vk::raii::Queue graphicsQueue = nullptr;
      vk::raii::Queue presentQueue = nullptr;
      vk::raii::SwapchainKHR swapChain = nullptr;
      std::vector<vk::Image> swapChainImages;
      vk::Format swapChainImageFormat;
      VkExtent2D swapChainExtent;
      std::vector<vk::raii::ImageView> swapChainImageViews;
      vk::raii::RenderPass renderPass = nullptr;
      vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
      vk::raii::PipelineLayout pipelineLayout = nullptr;
      vk::raii::Pipeline graphicsPipeline = nullptr;
      vk::raii::Pipeline meshPipeline = nullptr;
      std::vector<vk::raii::Framebuffer> swapChainFramebuffers;
      vk::raii::CommandPool commandPool = nullptr;
      vk::raii::DescriptorPool descriptorPool = nullptr;
      std::vector<FrameData> frameDatas;
      vk::SpecializationInfo specInfo;
      std::vector<vk::SpecializationMapEntry> mapEntries;
      const SpecializationData specData = {NUM_WAVES, OCEAN_WIDTH,
                                           OCEAN_CELL_WIDTH};
      vk::raii::Image depthImage = nullptr;
      vk::raii::ImageView depthImageView = nullptr;
      vk::raii::DeviceMemory depthImageMemory = nullptr;


      Wave waves[NUM_WAVES];

      std::default_random_engine gen;
      std::uniform_real_distribution<> distribution{0.0, 1.0};

      glm::vec3 cameraPos{50.0, -70.0, -80.0};
      glm::vec3 cameraFront{0.0, 0.0, 0.0};
      glm::vec3 cameraUp{0.0f, -1.0f, 0.0f};

      double lastX = WIDTH / 2.0;
      double lastY = HEIGHT / 2.0;
      bool firstMouse = true;
      float yaw = 90.0f;
      float pitch = 40.0f;
      float mouseSensitivity = 0.1f;

      const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
      int currentFrame = 0;

      float time = 0;
      bool running = true;

      GLFWwindow* initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        return window;
      }

      std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(
            glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
          extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
      }

      bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
          bool layerFound = false;

          for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
              layerFound = true;
              break;
            }
          }

          if (!layerFound) {
            return false;
          }
        }

        return true;
      }

      void createInstance() {
        vk::ApplicationInfo appInfo("vkEngine", VK_MAKE_API_VERSION(0, 1, 3, 0),
                                    "No Engine", VK_MAKE_API_VERSION(0, 1, 0, 0),
                                    VK_API_VERSION_1_3);
        auto extensions = getRequiredExtensions();

#ifdef NDEBUG
        vk::StructureChain<vk::InstanceCreateInfo> instanceCreateInfoChain(
            {{}, &appInfo, {}, extensions});
#else
    if (!checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }
    vk::StructureChain<vk::InstanceCreateInfo,
                       vk::DebugUtilsMessengerCreateInfoEXT>
        instanceCreateInfoChain({{}, &appInfo, validationLayers, extensions},
                                makeDebugUtilsMessengerCreateInfoEXT());
#endif

        vk::raii::Context context;
        instance = context.createInstance(
            instanceCreateInfoChain.get<vk::InstanceCreateInfo>());
      }

      vk::DebugUtilsMessengerCreateInfoEXT
      makeDebugUtilsMessengerCreateInfoEXT() {
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

        return {{}, severityFlags, messageTypeFlags, &debugCallback};
      }

      void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        debugMessenger = instance.createDebugUtilsMessengerEXT(
            makeDebugUtilsMessengerCreateInfoEXT());
      }

      void createSurface() {
        VkSurfaceKHR _surface;
        glfwCreateWindowSurface(*instance, window, nullptr, &_surface);
        surface = vk::raii::SurfaceKHR(instance, _surface);
      }

      QueueFamilyIndices findQueueFamilies(vk::raii::PhysicalDevice device) {
        QueueFamilyIndices indices;

        std::vector<vk::QueueFamilyProperties> queueFamilies =
            device.getQueueFamilyProperties();

        int i = 0;
        for (const vk::QueueFamilyProperties& queueFamily : queueFamilies) {
          if (device.getSurfaceSupportKHR(i, surface)) {
            indices.presentFamily = i;
          }
          if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
          }

          if (indices.isComplete()) {
            break;
          }

          i++;
        }

        return indices;
      }

      bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice device) {
        std::vector<vk::ExtensionProperties> availableExtensions =
            device.enumerateDeviceExtensionProperties();

        std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                                 deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
          requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
      }

      bool isDeviceSuitable(vk::raii::PhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
          SwapChainSupportDetails swapChainSupport =
              querySwapChainSupport(device);
          swapChainAdequate = !swapChainSupport.formats.empty() &&
                              !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
      }

      void pickPhysicalDevice() {
        vk::raii::PhysicalDevices physicalDevices(instance);
        for (vk::raii::PhysicalDevice device : physicalDevices) {
          if (isDeviceSuitable(device)) {
            physicalDevice = device;
            return;
          }
        }

        throw std::runtime_error("failed to find a suitable GPU!");
      }

      void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(), indices.presentFamily.value()};

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
          vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1,
                                                    &queuePriority);
          queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures{};
        vk::PhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{true, true};

#ifdef NDEBUG
        device = physicalDevice.createDevice({{},
                                              queueCreateInfos,
                                              {},
                                              deviceExtensions,
                                              &deviceFeatures,
                                              &meshShaderFeatures});
#else
        device = physicalDevice.createDevice({{},
                                          queueCreateInfos,
                                          validationLayers,
                                          deviceExtensions,
                                          &deviceFeatures,
                                          &meshShaderFeatures});
#endif
        
        auto deviceProps2 = physicalDevice.getProperties2<
          vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMeshShaderPropertiesEXT>();
        vk::PhysicalDeviceMeshShaderPropertiesEXT const& meshShaderProperties =
            deviceProps2.get<vk::PhysicalDeviceMeshShaderPropertiesEXT>();
        uint32_t maxMeshWorkGroupTotalCount = meshShaderProperties.maxMeshWorkGroupTotalCount;
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
      }

      struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
      };

      SwapChainSupportDetails querySwapChainSupport(
          vk::raii::PhysicalDevice device) {
        vk::SurfaceCapabilitiesKHR capabilities =
            device.getSurfaceCapabilitiesKHR(surface);
        std::vector<vk::SurfaceFormatKHR> formats =
            device.getSurfaceFormatsKHR(surface);
        std::vector<vk::PresentModeKHR> presentModes =
            device.getSurfacePresentModesKHR(surface);

        return {capabilities, formats, presentModes};
      }

      vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
          const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
          if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
              availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
          }
        }
        return availableFormats[0];
      }

      vk::PresentModeKHR chooseSwapPresentMode(
          const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
          if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
          }
        }

        return vk::PresentModeKHR::eFifo;
      }

      vk::Extent2D chooseSwapExtent(
          const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max()) {
          return capabilities.currentExtent;
        } else {
          int width, height;
          glfwGetFramebufferSize(window, &width, &height);

          vk::Extent2D actualExtent(width, height);

          actualExtent.width =
              std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                         capabilities.maxImageExtent.width);
          actualExtent.height = std::clamp(actualExtent.height,
                                           capabilities.minImageExtent.height,
                                           capabilities.maxImageExtent.height);

          return actualExtent;
        }
      }

      void createSwapChain() {
        SwapChainSupportDetails swapChainSupport =
            querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat =
            chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode =
            chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount) {
          imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                         indices.presentFamily.value()};

        vk::SharingMode imageSharingMode;
        if (indices.graphicsFamily != indices.presentFamily) {
          imageSharingMode = vk::SharingMode::eConcurrent;
        } else {
          imageSharingMode = vk::SharingMode::eExclusive;
        }

        vk::SwapchainCreateInfoKHR swapchainCreateInfo(
            {}, surface, imageCount, surfaceFormat.format,
            surfaceFormat.colorSpace, extent, 1,
            vk::ImageUsageFlagBits::eColorAttachment, imageSharingMode, 2,
            queueFamilyIndices, swapChainSupport.capabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, true);

        swapChain = device.createSwapchainKHR(swapchainCreateInfo);
        swapChainImages = swapChain.getImages();
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
      }

      void createImageViews() {
        swapChainImageViews.clear();
        for (vk::Image image : swapChainImages) {
          swapChainImageViews.push_back(device.createImageView(
              {{},
               image,
               vk::ImageViewType::e2D,
               swapChainImageFormat,
               {vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity},
               {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}));
        }
      }

      void createRenderPass() {
        vk::AttachmentDescription colorAttachment(
            {}, swapChainImageFormat, vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentReference colorAttachmentRef(
            0, vk::ImageLayout::eColorAttachmentOptimal);

        vk::AttachmentDescription depthAttachment(
            {}, findDepthFormat(), vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::AttachmentReference depthAttachmentRef(
            1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::SubpassDescription subpass;
        subpass.setColorAttachments(colorAttachmentRef);
        subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
        subpass.setPDepthStencilAttachment(&depthAttachmentRef);

        vk::SubpassDependency subpassDependency(
            vk::SubpassExternal, 0,
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eLateFragmentTests,
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
            vk::AccessFlagBits::eNone | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            {});

        std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment,depthAttachment};
        vk::RenderPassCreateInfo renderPassCreateInfo;
        renderPassCreateInfo.setAttachments(attachments);
        renderPassCreateInfo.setSubpasses(subpass);
        renderPassCreateInfo.setDependencies(subpassDependency);

        renderPass = device.createRenderPass(renderPassCreateInfo);
      }

      static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
          throw std::runtime_error("failed to open file!");
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
      }

      vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) {
        return device.createShaderModule(
            {{}, code.size(), reinterpret_cast<const uint32_t*>(code.data())});
      }

      void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding layoutBinding(
            0, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eMeshEXT |
            vk::ShaderStageFlagBits::eFragment,
            nullptr);
        descriptorSetLayout =
            device.createDescriptorSetLayout({{}, layoutBinding});
      }

      void createSpecialization() {
        mapEntries.push_back(
            {0, offsetof(SpecializationData, num_waves), sizeof(int)});

        mapEntries.push_back(
            {1, offsetof(SpecializationData, ocean_width), sizeof(int)});

        mapEntries.push_back(
            {2, offsetof(SpecializationData, ocean_cell_width), sizeof(float)});

        specInfo = {uint32_t(mapEntries.size()), mapEntries.data(), sizeof(specData), &specData};
      }

      void createMeshPipeline() {
        auto meshShaderCode = readFile("shaders/mesh.spv");
        auto taskShaderCode = readFile("shaders/task.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        vk::raii::ShaderModule meshShaderModule =
            createShaderModule(meshShaderCode);
        vk::raii::ShaderModule taskShaderModule =
            createShaderModule(taskShaderCode);
        vk::raii::ShaderModule fragShaderModule =
            createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo taskShaderStageInfo(
            {}, vk::ShaderStageFlagBits::eTaskEXT, taskShaderModule, "main", &specInfo);
        vk::PipelineShaderStageCreateInfo meshShaderStageInfo(
            {}, vk::ShaderStageFlagBits::eMeshEXT, meshShaderModule, "main", &specInfo);
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
            {}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main", &specInfo);

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
            taskShaderStageInfo, meshShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        //vk::PipelineVertexInputStateCreateInfo vertexInputState(
        //    {}, bindingDescription, attributeDescriptions);

        //vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState(
        //    {}, vk::PrimitiveTopology::eTriangleList, false);

        vk::Viewport viewport(0.0f, 0.0f, swapChainExtent.width,
                              swapChainExtent.height, 0.0f, 1.0f);

        vk::Rect2D scissor({0, 0}, swapChainExtent);

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

        vk::PipelineViewportStateCreateInfo viewportState({}, viewport,
                                                          scissor);

        vk::PipelineRasterizationStateCreateInfo rasterizationState(
            {}, false, false, vk::PolygonMode::eFill,
            vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
            false, 0.0f, 0.0f, 0.0f, 1.0f);

        vk::PipelineMultisampleStateCreateInfo multisampleState(
            {}, vk::SampleCountFlagBits::e1, false);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.setColorWriteMask(
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        colorBlendAttachment.setBlendEnable(false);

        vk::PipelineColorBlendStateCreateInfo colorBlendState(
            {}, false, vk::LogicOp::eCopy, colorBlendAttachment);

        
        vk::PipelineDepthStencilStateCreateInfo depthStencilState;
        depthStencilState.setDepthTestEnable(true);
        depthStencilState.setDepthWriteEnable(true);
        depthStencilState.setDepthCompareOp(vk::CompareOp::eLess);
        depthStencilState.setDepthBoundsTestEnable(false);
        depthStencilState.setStencilTestEnable(false);

        vk::PushConstantRange push_constant;
        push_constant.offset = 0;
        push_constant.size = sizeof(PushConstants);
        push_constant.stageFlags = vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment;

        const vk::PipelineLayoutCreateInfo plci{
            {}, *descriptorSetLayout,
            push_constant
        };
        pipelineLayout = device.createPipelineLayout(plci);


        vk::GraphicsPipelineCreateInfo pipelineCreateInfo(
            {}, shaderStages, nullptr, nullptr, nullptr, &viewportState,
            &rasterizationState, &multisampleState, &depthStencilState,
            &colorBlendState, &dynamicState, pipelineLayout, renderPass, 0);

        meshPipeline =
            device.createGraphicsPipeline(nullptr, pipelineCreateInfo);
      }

      void createFramebuffers() {
        swapChainFramebuffers.clear();
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
          std::array<vk::ImageView, 2> attachments = {swapChainImageViews[i],
                                                      depthImageView};
          vk::FramebufferCreateInfo framebufferInfo;
          framebufferInfo.setAttachments(attachments);
          framebufferInfo.setRenderPass(*renderPass);
          framebufferInfo.setWidth(swapChainExtent.width);
          framebufferInfo.setHeight(swapChainExtent.height);
          framebufferInfo.setLayers(1);
          swapChainFramebuffers.push_back(
              device.createFramebuffer(framebufferInfo));
        }
      }

      vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
                                   vk::FormatFeatureFlags features) {
        for (vk::Format format : candidates) {
          vk::FormatProperties props =
              physicalDevice.getFormatProperties(format);
          if (props.optimalTilingFeatures & features) {
            return format;
          }
        }
        throw std::runtime_error("failed to find supported format!");
      }

      vk::Format findDepthFormat() {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32Sfloat,
             vk::Format::eD24UnormS8Uint},
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);
      }

      void createDepthResources() {
        vk::Format format = findDepthFormat();

        vk::ImageCreateInfo imageCreateInfo{
          {},
          vk::ImageType::e2D,
            format,
            {swapChainExtent.width, swapChainExtent.height, 1},
            1,      1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::SharingMode::eExclusive
        };
        depthImage = device.createImage(imageCreateInfo);

        vk::MemoryRequirements memRequirements = depthImage.getMemoryRequirements();

        depthImageMemory = device.allocateMemory(
            {memRequirements.size,
             findMemoryType(memRequirements.memoryTypeBits,
                            vk::MemoryPropertyFlagBits::eDeviceLocal)});

        vk::BindImageMemoryInfo bindInfo{*depthImage, *depthImageMemory, 0};
        device.bindImageMemory2(bindInfo);

        depthImageView = device.createImageView(
            {{},
             depthImage,
             vk::ImageViewType::e2D,
             format,
             {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
              vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
             {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}});

      }

      void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices =
            findQueueFamilies(physicalDevice);
        commandPool = device.createCommandPool(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
             queueFamilyIndices.graphicsFamily.value()});
      }

      uint32_t findMemoryType(uint32_t typeFilter,
                              vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties =
            physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
          if (typeFilter & (1 << i) &&
              (memProperties.memoryTypes[i].propertyFlags & properties) ==
                  properties) {
            return i;
          }
        }

        throw std::runtime_error("failed to find suitable memory type!");
      }

      BufferData createBuffer(vk::DeviceSize bufferSize,
                              vk::BufferUsageFlags usage,
                              vk::MemoryPropertyFlags properties) {
        vk::raii::Buffer buffer(
            device, {{}, bufferSize, usage, vk::SharingMode::eExclusive});

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

        uint32_t memoryType =
            findMemoryType(memRequirements.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eHostVisible |
                               vk::MemoryPropertyFlagBits::eHostCoherent);

        vk::raii::DeviceMemory memory(device,
                                      {memRequirements.size, memoryType});

        buffer.bindMemory(memory, 0);

        return BufferData(buffer, memory, (void*)nullptr);
      }

      BufferData createUniformBufferWaves() {
        vk::DeviceSize bufferSize = sizeof(WavesUBO);
        BufferData bufferData =
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostCoherent |
                             vk::MemoryPropertyFlagBits::eHostVisible);
        bufferData.mapped = bufferData.memory.mapMemory(0, bufferSize);
        return bufferData;
      }

      void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                      vk::DeviceSize size) {
        vk::raii::CommandBuffer cb = std::move(device.allocateCommandBuffers(
            {commandPool, vk::CommandBufferLevel::ePrimary, 1})[0]);

        vk::BufferCopy region(0, 0, size);

        cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb.copyBuffer(srcBuffer, dstBuffer, region);
        cb.end();

        vk::SubmitInfo submitInfo({}, 0, *cb);
        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();
      }

      void recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer,
                               uint32_t imageIndex, uint32_t currentFrame,
                               ImDrawData* draw_data) {
        commandBuffer.begin({});

        vk::Rect2D renderArea({{0, 0}, swapChainExtent});
        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].setColor({0.0f, 0.0f, 0.0f, 1.0f});
        clearValues[1].setDepthStencil({1.0f, 0});
        vk::RenderPassBeginInfo renderPassInfo;
        renderPassInfo.setClearValues(clearValues);
        renderPassInfo.setRenderArea(renderArea);
        renderPassInfo.setRenderPass(*renderPass);
        renderPassInfo.setFramebuffer(*swapChainFramebuffers[imageIndex]);

        commandBuffer.beginRenderPass(renderPassInfo,
                                      vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                   meshPipeline);

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
            *frameDatas[currentFrame].descriptorSet, {});

        vk::Viewport viewport(0.0f, 0.0f, swapChainExtent.width,
                              swapChainExtent.height, 0.0f, 1.0f);

        commandBuffer.setViewport(0, viewport);
        commandBuffer.setScissor(0, renderArea);
        commandBuffer.pushConstants<PushConstants>(pipelineLayout,
            vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
            0,frameDatas[currentFrame].pushConstants);

        commandBuffer.drawMeshTasksEXT(1,1,1);

        ImGui_ImplVulkan_RenderDrawData(draw_data, *commandBuffer);

        commandBuffer.endRenderPass();
        commandBuffer.end();
      }

      void createDescriptorPool() {
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
        descriptorPool = device.createDescriptorPool(
            {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1000, poolsize});
      }

      void createFrameData() {
        vk::raii::CommandBuffers commandBuffers(
            device, {commandPool, vk::CommandBufferLevel::ePrimary,
                     MAX_FRAMES_IN_FLIGHT});

        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                     descriptorSetLayout);
        vk::raii::DescriptorSets descriptorSets(device,
                                                {descriptorPool, layouts});

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
          vk::raii::Semaphore imageAvailableSemaphore(device, {{}, nullptr});
          vk::raii::Semaphore renderFinishedSemaphore(device, {{}, nullptr});
          vk::raii::Fence inFlightFence(device,
                                        {vk::FenceCreateFlagBits::eSignaled});
          BufferData uniformBuffer = createUniformBufferWaves();

          vk::DescriptorBufferInfo info(uniformBuffer.buffer, 0,
                                        sizeof(WavesUBO));

          vk::WriteDescriptorSet descriptorWrite(
              descriptorSets[i], 0, 0, vk::DescriptorType::eUniformBuffer, {},
              info);

          device.updateDescriptorSets(descriptorWrite, {});

          FrameData frameData(
              {std::move(commandBuffers[i]), std::move(imageAvailableSemaphore),
                               std::move(renderFinishedSemaphore),
                               std::move(inFlightFence), 
               std::move(uniformBuffer),
                               {}, std::move(descriptorSets[i])});
          frameDatas.push_back(std::move(frameData));
        }
      }

      void initImGui() {
        //vk::raii::DescriptorPool pool_sizes[] = {
        //  {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        //  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        //  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        //  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        //  {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        //  {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        //  {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        //  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        //  {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        //  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        //  {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
        //VkDescriptorPoolCreateInfo pool_info = {};
        //pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        //pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        //pool_info.maxSets = 1000;
        //pool_info.poolSizeCount = std::size(pool_sizes);
        //pool_info.pPoolSizes = pool_sizes;

        //VkDescriptorPool imguiPool;
        //vkCreateDescriptorPool(*device, &pool_info, nullptr, &imguiPool);

        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForVulkan(window, true);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.ApiVersion = VK_API_VERSION_1_3;
        init_info.Instance = *instance;
        init_info.RenderPass = *renderPass;
        init_info.PhysicalDevice = *physicalDevice;
        init_info.Device = *device;
        init_info.Queue = *graphicsQueue;
        init_info.DescriptorPool = *descriptorPool;
        init_info.Subpass = 0;
        init_info.MinImageCount = 2;
        init_info.ImageCount = 2;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        ImGui_ImplVulkan_Init(&init_info);
      }

      void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createSpecialization();
        createMeshPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createDescriptorPool();
        createFrameData();
        createWaves();
      }
      void processInput(GLFWwindow* window) {
        glm::vec3 front;
        front.x =
            cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z =
            sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);
        float cameraSpeed = 0.05f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
          cameraPos += cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
          cameraPos -= cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
          cameraPos -=
              glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
          cameraPos +=
              glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
          running = !running;
      }

      void updatePushConstants(FrameData& fd) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        if (running) {
          time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();
        }
        
        
        fd.pushConstants.time = time;
        fd.pushConstants.view =
            glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        fd.pushConstants.proj = glm::perspective(
            glm::radians(45.0f),
            swapChainExtent.width / (float)swapChainExtent.height, 0.1f,
            1000.0f);
        fd.pushConstants.proj[1][1] *= -1;
        fd.pushConstants.cameraPos = glm::vec4(cameraPos, 1.0);
      }

      float rf() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        return dis(gen);
      }

      void createWaves() {
        WavesUBO ubo{};
        float amp_factor = 0.78f;
        float freq_factor = 1.18;
        float sharpness_factor = 1.05;
        float amp = .3f;
        float freq = 0.1f;
        float sharpness = 1.2f;
        for (int i = 0; i < NUM_WAVES; i++) {
          Wave wave = {};
          
          glm::vec2 dir(rf() * 1.5 - 1, rf() * 1.5 - 1);
          dir = glm::normalize(dir);
          wave.dir_x = dir.x;
          wave.dir_z = dir.y;
          wave.freq = freq;
          wave.amp = amp;
          wave.phase = rf()*2+0.5;
          wave.sharpness = sharpness;
          waves[i] = wave;

          amp *= amp_factor;
          freq *= freq_factor;
          sharpness *= sharpness_factor;
        }
      }

      void updateWaves(FrameData& fd) {
        WavesUBO ubo{};
        for (int i = 0; i < NUM_WAVES; i++) {
          ubo.waves[i] = waves[i];
        }
        memcpy(fd.wavesBuffer.mapped, &ubo, sizeof(ubo));
      }

      void cleanupSwapChain() { swapChain = nullptr; }

      void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
          glfwWaitEvents();
          glfwGetFramebufferSize(window, &width, &height);
        }

        device.waitIdle();
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createFramebuffers();
      }

      void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
      }
    };

    int main() {
      Engine engine;

      try {
        engine.run();
      } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
      }

      return EXIT_SUCCESS;
    }
