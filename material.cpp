#include "material.h"
#include "volkan.h"

#include <vulkan/vulkan_raii.hpp>

float rf() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  return dis(gen);
}


Waves createWaves() {
  float amp_factor = 0.78f;
  float freq_factor = 1.18;
  float sharpness_factor = 1.05;
  float amp = .3f;
  float freq = 0.1f;
  float sharpness = 1.2f;

  Waves waves;
  for (int i = 0; i < 10; i++) {
    Wave& wave = waves.waves[i];

    glm::vec2 dir(rf() * 1.5 - 1, rf() * 1.5 - 1);
    dir = glm::normalize(dir);
    wave.dir_x = dir.x;
    wave.dir_z = dir.y;
    wave.freq = freq;
    wave.amp = amp;
    wave.phase = rf() * 2 + 0.5;
    wave.sharpness = sharpness;

    amp *= amp_factor;
    freq *= freq_factor;
    sharpness *= sharpness_factor;
  }
  return waves;
}

WaterMaterial::WaterMaterial(Volkan& volkan, SpecData specData, vk::raii::RenderPass& renderPass) : Material(){
  // Shaders
  vk::raii::ShaderModule meshShaderModule =
      volkan.createShaderModule("shaders/mesh.spv");
  vk::raii::ShaderModule taskShaderModule =
      volkan.createShaderModule("shaders/task.spv"); 
  vk::raii::ShaderModule fragShaderModule =
      volkan.createShaderModule("shaders/frag.spv");

  // Specialization constants
  const std::vector<vk::SpecializationMapEntry> specMapEntries = SpecData::specMapEntries();
  vk::SpecializationInfo specInfo;
  specInfo.setMapEntries(specMapEntries);
  specInfo.setData<SpecData>(specData);

  vk::PipelineShaderStageCreateInfo taskShaderStageInfo;
  taskShaderStageInfo.setStage(vk::ShaderStageFlagBits::eTaskEXT);
  taskShaderStageInfo.setModule(*taskShaderModule);
  taskShaderStageInfo.setPName("main");
  taskShaderStageInfo.setPSpecializationInfo(&specInfo);

  vk::PipelineShaderStageCreateInfo meshShaderStageInfo;
  meshShaderStageInfo.setStage(vk::ShaderStageFlagBits::eMeshEXT);
  meshShaderStageInfo.setModule(*meshShaderModule);
  meshShaderStageInfo.setPName("main");
  meshShaderStageInfo.setPSpecializationInfo(&specInfo);

  vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
  fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
  fragShaderStageInfo.setModule(*fragShaderModule);
  fragShaderStageInfo.setPName("main");
  fragShaderStageInfo.setPSpecializationInfo(&specInfo);

  std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
      taskShaderStageInfo, meshShaderStageInfo, fragShaderStageInfo};

  vk::Viewport viewport = volkan.getViewport();
  vk::Rect2D scissor = volkan.getScissor();

  std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                 vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

  vk::PipelineViewportStateCreateInfo viewportState({}, viewport, scissor);

  vk::PipelineRasterizationStateCreateInfo rasterizationState(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

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
  push_constant.size = sizeof(PerFrameVariables);
  push_constant.stageFlags =
      vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment;

  vk::DescriptorSetLayoutBinding layoutBinding;
  layoutBinding.setBinding(0);
  layoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
  layoutBinding.setDescriptorCount(1);
  layoutBinding.setStageFlags(vk::ShaderStageFlagBits::eMeshEXT |
                              vk::ShaderStageFlagBits::eFragment);

  vk::DescriptorSetLayoutCreateInfo layoutInfo;
  layoutInfo.setBindings(layoutBinding);
  descriptorSetLayout = volkan.createDescriptorSetLayout(layoutInfo);

  descriptorSets = volkan.createDescriptorSets(descriptorSetLayout);

  
  Waves waves = createWaves();
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    wavesBuffers.push_back(volkan.allocateUniformBuffer(descriptorSets[i], sizeof(Waves)));
    memcpy(wavesBuffers[i].mapped, &waves, sizeof(Waves));
  }

  vk::PipelineLayoutCreateInfo plci;
  plci.setSetLayouts(*descriptorSetLayout);
  plci.setPushConstantRanges(push_constant);
  pipelineLayout = volkan.createPipelineLayout(plci);

  vk::GraphicsPipelineCreateInfo pipelineCI;
  pipelineCI.setStages(shaderStages);
  pipelineCI.setPViewportState(&viewportState);
  pipelineCI.setPRasterizationState(&rasterizationState);
  pipelineCI.setPMultisampleState(&multisampleState);
  pipelineCI.setPDepthStencilState(&depthStencilState);
  pipelineCI.setPColorBlendState(&colorBlendState);
  pipelineCI.setPDynamicState(&dynamicState);
  pipelineCI.setLayout(*pipelineLayout);
  pipelineCI.setRenderPass(*renderPass);
  pipelineCI.setSubpass(0);

  pipeline = volkan.createPipeline(pipelineCI);
}

// RayTracingMaterial implementation
RayTracingMaterial::RayTracingMaterial(Volkan& volkan,
                                       vk::raii::AccelerationStructureKHR& tlas,
                                       vk::raii::ImageView& storageImageView) : Material() {
  // Create descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings = {
    {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},
    {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}
  };

  vk::DescriptorSetLayoutCreateInfo layoutInfo;
  layoutInfo.setBindings(bindings);
  descriptorSetLayout = volkan.createDescriptorSetLayout(layoutInfo);

  // Allocate descriptor sets
  descriptorSets = volkan.createDescriptorSets(descriptorSetLayout);

  // Update descriptor sets (for both frames)
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::WriteDescriptorSetAccelerationStructureKHR asInfo;
    asInfo.setAccelerationStructures(*tlas);

    vk::WriteDescriptorSet asWrite;
    asWrite.setDstSet(*descriptorSets[i]);
    asWrite.setDstBinding(0);
    asWrite.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
    asWrite.setDescriptorCount(1);
    asWrite.setPNext(&asInfo);

    vk::DescriptorImageInfo imageInfo;
    imageInfo.setImageView(*storageImageView);
    imageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet imageWrite;
    imageWrite.setDstSet(*descriptorSets[i]);
    imageWrite.setDstBinding(1);
    imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    imageWrite.setImageInfo(imageInfo);

    std::vector<vk::WriteDescriptorSet> writes = {asWrite, imageWrite};
    volkan.device_.updateDescriptorSets(writes, {});
  }

  // Create pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayoutInfo.setSetLayouts(*descriptorSetLayout);
  pipelineLayout = volkan.createPipelineLayout(pipelineLayoutInfo);

  // Load shaders
  vk::raii::ShaderModule raygenShader = volkan.createShaderModule("shaders/raygen.spv");
  vk::raii::ShaderModule missShader = volkan.createShaderModule("shaders/miss.spv");
  vk::raii::ShaderModule chitShader = volkan.createShaderModule("shaders/closesthit.spv");

  // Shader stages
  std::vector<vk::PipelineShaderStageCreateInfo> stages = {
    {{}, vk::ShaderStageFlagBits::eRaygenKHR, *raygenShader, "main"},
    {{}, vk::ShaderStageFlagBits::eMissKHR, *missShader, "main"},
    {{}, vk::ShaderStageFlagBits::eClosestHitKHR, *chitShader, "main"}
  };

  // Shader groups
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> groups = {
    {vk::RayTracingShaderGroupTypeKHR::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
    {vk::RayTracingShaderGroupTypeKHR::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
    {vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR}
  };

  // Create ray tracing pipeline
  vk::RayTracingPipelineCreateInfoKHR pipelineInfo;
  pipelineInfo.setStages(stages);
  pipelineInfo.setGroups(groups);
  pipelineInfo.setMaxPipelineRayRecursionDepth(1);
  pipelineInfo.setLayout(*pipelineLayout);

  pipeline = volkan.createRayTracingPipeline(pipelineInfo);

  // Create SBT
  auto rtProps = volkan.getRayTracingProperties();
  uint32_t handleSize = rtProps.shaderGroupHandleSize;
  uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
  uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;

  auto alignUp = [](uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
  };

  uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

  // Get shader group handles
  uint32_t groupCount = 3;
  size_t sbtSize = groupCount * handleSizeAligned;
  std::vector<uint8_t> handles = pipeline.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, groupCount, sbtSize);

  // Create SBT buffers
  raygenSBT_ = volkan.createBuffer(
    alignUp(handleSizeAligned, baseAlignment),
    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );

  missSBT_ = volkan.createBuffer(
    alignUp(handleSizeAligned, baseAlignment),
    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );

  hitSBT_ = volkan.createBuffer(
    alignUp(handleSizeAligned, baseAlignment),
    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );

  // Copy handles to SBT buffers
  void* raygenData = raygenSBT_.memory.mapMemory(0, handleSizeAligned);
  memcpy(raygenData, handles.data(), handleSize);
  raygenSBT_.memory.unmapMemory();

  void* missData = missSBT_.memory.mapMemory(0, handleSizeAligned);
  memcpy(missData, handles.data() + handleSizeAligned, handleSize);
  missSBT_.memory.unmapMemory();

  void* hitData = hitSBT_.memory.mapMemory(0, handleSizeAligned);
  memcpy(hitData, handles.data() + 2 * handleSizeAligned, handleSize);
  hitSBT_.memory.unmapMemory();
}

void RayTracingMaterial::traceRays(vk::raii::CommandBuffer& commandBuffer,
                                   uint32_t width, uint32_t height, Volkan& volkan,
                                   size_t currentFrameIndex) {
  auto rtProps = volkan.getRayTracingProperties();
  uint32_t handleSize = rtProps.shaderGroupHandleSize;
  uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
  uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;

  auto alignUp = [](uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
  };

  uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);
  uint32_t alignedSize = alignUp(handleSizeAligned, baseAlignment);

  // Get buffer addresses
  vk::BufferDeviceAddressInfo raygenAddrInfo;
  raygenAddrInfo.setBuffer(*raygenSBT_.buffer);
  vk::DeviceAddress raygenAddr = volkan.device_.getBufferAddress(raygenAddrInfo);

  vk::BufferDeviceAddressInfo missAddrInfo;
  missAddrInfo.setBuffer(*missSBT_.buffer);
  vk::DeviceAddress missAddr = volkan.device_.getBufferAddress(missAddrInfo);

  vk::BufferDeviceAddressInfo hitAddrInfo;
  hitAddrInfo.setBuffer(*hitSBT_.buffer);
  vk::DeviceAddress hitAddr = volkan.device_.getBufferAddress(hitAddrInfo);

  // Setup SBT regions
  vk::StridedDeviceAddressRegionKHR raygenRegion;
  raygenRegion.setDeviceAddress(raygenAddr);
  raygenRegion.setStride(alignedSize);
  raygenRegion.setSize(alignedSize);

  vk::StridedDeviceAddressRegionKHR missRegion;
  missRegion.setDeviceAddress(missAddr);
  missRegion.setStride(alignedSize);
  missRegion.setSize(alignedSize);

  vk::StridedDeviceAddressRegionKHR hitRegion;
  hitRegion.setDeviceAddress(hitAddr);
  hitRegion.setStride(alignedSize);
  hitRegion.setSize(alignedSize);

  vk::StridedDeviceAddressRegionKHR callableRegion{};

  // Bind pipeline and descriptor sets
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0,
                                   *descriptorSets[currentFrameIndex], {});

  // Trace rays
  commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, callableRegion,
                             width, height, 1);
}