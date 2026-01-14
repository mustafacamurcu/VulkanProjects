#pragma once
#include <vulkan/vulkan_raii.hpp>

#include "vk_utils.h"
#include "structs.h"

class Volkan;

class Material {
 public:
  Material() = default;
  virtual ~Material() = default;

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  std::vector<BufferData> wavesBuffers;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;
};


class WaterMaterial : public Material {
 public:
  struct SpecData {
    int num_waves = 10;
    int ocean_width = 200;
    float ocean_cell_width = 1.0f;

    static std::vector<vk::SpecializationMapEntry> specMapEntries() {
      return {
          {0, offsetof(SpecData, num_waves), sizeof(int)},
          {1, offsetof(SpecData, ocean_width), sizeof(int)},
          {2, offsetof(SpecData, ocean_cell_width), sizeof(float)},
      };
    }
  };

  WaterMaterial(Volkan& volkan, SpecData specData,
                vk::raii::RenderPass& renderPass);
 private:
};

class RayTracingMaterial : public Material {
 public:
  RayTracingMaterial(Volkan& volkan,
                     vk::raii::AccelerationStructureKHR& tlas,
                     vk::raii::ImageView& storageImageView);

  void traceRays(vk::raii::CommandBuffer& commandBuffer,
                 uint32_t width, uint32_t height, Volkan& volkan,
                 size_t currentFrameIndex);

 private:
  BufferData raygenSBT_;
  BufferData missSBT_;
  BufferData hitSBT_;
};