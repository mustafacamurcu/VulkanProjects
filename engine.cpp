#include "engine.h"
#include <iostream>
#include <glm/glm.hpp>
#include <random>

#include "vk_utils.h"

#include "material.h"

static void framebufferResizeCallback(GLFWwindow* window, int width,
                                      int height) {
  auto engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  engine->handleFramebufferResized();
}

static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
  Engine* engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  engine->mouseEvent(xpos, ypos);
}

void Engine::createWaterPipeline() {
  waterMaterial_ = std::make_unique<WaterMaterial>(
      volkan_, WaterMaterial::SpecData{10, 200, 1.0f}, volkan_.renderPass_);
}

Engine::Engine() {
  createWindow(WIDTH, HEIGHT, "Vulkan Scene");
  volkan_.init(window_);
  createWaterPipeline();
}

Engine::~Engine() {
  waterMaterial_.reset();
  volkan_.cleanupRenderResources();
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Engine::run() {
  while (!glfwWindowShouldClose(window_)) {

    glfwPollEvents();
    processInput(window_);
    
    drawFrame();
  }
}

void Engine::drawFrame() {
  static auto startTime = std::chrono::high_resolution_clock::now();
  auto currentTime = std::chrono::high_resolution_clock::now();

  float time = std::chrono::duration<float, std::chrono::seconds::period>(
                   currentTime - startTime)
                   .count();
  if (!volkan_.nextFrameReady()) {
    return;
  }

  volkan_.beginCommandBuffer();
  volkan_.beginRenderPass();
  volkan_.bindPipeline(camera_, time, *waterMaterial_);
  volkan_.drawMesh();
  volkan_.submit();

}

void Engine::processInput(GLFWwindow* window) {
  glm::vec3 front;
  front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  front.y = sin(glm::radians(pitch));
  front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
  camera_.front = glm::normalize(front);
  float cameraSpeed = 0.05f;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera_.pos += cameraSpeed * camera_.front;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera_.pos -= cameraSpeed * camera_.front;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera_.pos -=
        glm::normalize(glm::cross(camera_.front, camera_.up)) * cameraSpeed;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera_.pos +=
        glm::normalize(glm::cross(camera_.front, camera_.up)) * cameraSpeed;
}

void Engine::createWindow(int width, int height, const char* title) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  window_ = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
  glfwSetCursorPosCallback(window_, mouseCallback);
  glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void Engine::handleFramebufferResized() {
  framebufferResized = true;
}

void Engine::mouseEvent(double xpos, double ypos) {
  // std::cout << "Mouse moved to: " << xpos << ", " << ypos << std::endl;
}


