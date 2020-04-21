#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    echo "./compile.sh VulkanT12"
    exit 1
fi

shaderFolder=$1/shaders

cd $shaderFolder

$VULKAN_SDK/bin/glslangValidator -V shader.vert
$VULKAN_SDK/bin/glslangValidator -V shader.frag
