#include "resource.hpp"

// Printing
std::string format_as(Resource::Type type)
{
	return "Float32";
}

std::string format_as(Resource::Device device)
{
	switch (device) {
	case Resource::eCPU:
		return "CPU";
	case Resource::eCUDA:
		return "CUDA";
	case Resource::eVulkan:
		return "Vulkan";
	default:
		break;
	}

	return "?";
}
