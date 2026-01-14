#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define NUM_WAVES 10

#define MAX_FRAMES_IN_FLIGHT 2

struct Camera {
  glm::vec3 pos{50.0, -70.0, -80.0};
  glm::vec3 front{0.0, 0.0, 0.0};
  glm::vec3 up{0.0f, -1.0f, 0.0f};

  glm::mat4 getViewMatrix() {
    return glm::lookAt(pos, pos + front, up); }

  glm::mat4 getProjectionMatrix(float aspectRatio) {
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
    proj[1][1] *= -1; // Invert Y for Vulkan
    return proj;
  }
};

  
struct PerFrameVariables {
  glm::float32 time;
  alignas(16) glm::mat4 view;
  glm::mat4 proj;
  glm::vec4 cameraPos;
};

struct alignas(32) Wave {
  float amp;
  float phase;
  float dir_x;
  float dir_z;
  float freq;
  float sharpness;
};

struct Waves {
  Wave waves[NUM_WAVES];
};