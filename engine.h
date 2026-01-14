#pragma once
#include <GLFW/glfw3.h>
#include "volkan.h"
#include "structs.h"
#include "material.h"



class Engine {
 public:
  Engine();
  ~Engine();

  void run();
  
  void mouseEvent(double xpos, double ypos);

  bool shouldClose() const;
  void pollEvents() const;
  void getFramebufferSize(int* width, int* height) const;

  void handleFramebufferResized();

  bool firstMouse = true;
  float lastX;
  float lastY;
  float mouseSensitivity = 0.1f;
  float yaw = 90.0f;
  float pitch = 40.0f;
  bool framebufferResized = false;

  const uint32_t WIDTH = 1920;
  const uint32_t HEIGHT = 1080;

 private:
  void createWindow(int width, int height, const char* title);

  void createWaterPipeline();
  void createRayTracingResources();

  void processInput(GLFWwindow* window);
  void drawFrame();

  std::unique_ptr<Material> waterMaterial_;
  std::unique_ptr<RayTracingMaterial> rtMaterial_;

  // Ray tracing geometry
  struct Vertex {
    float x, y, z;
  };
  BufferData triangleVertexBuffer_;
  vk::raii::AccelerationStructureKHR blas_ = nullptr;
  BufferData blasBuffer_;
  vk::raii::AccelerationStructureKHR tlas_ = nullptr;
  BufferData tlasBuffer_;

  GLFWwindow* window_;
  Volkan volkan_;

  Camera camera_;

};