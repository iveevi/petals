#pragma once

#include <cstring>
#include <optional>
#include <string>

struct Resource {
	// TODO: variant of all pointer types and vk buffer
	float *ptr = nullptr;
	bool owner = false;
	size_t elements = 0;

	enum Type {
		f32
	} type;

	enum Device {
		eCPU,
		eCUDA,
		eVulkan
	} device;

	// Memset each element
	void memset(float value) const {
		for (size_t i = 0; i < elements; i++)
			ptr[i] = value;
	}

	// Slices are not owners
	std::optional <Resource> slice(long int start = 0, long int end = -1) const {
		// NOTE: End is not inclusive
		if (start >= elements)
			return std::nullopt;
		if (end < 0)
			end = elements;

		return Resource {
			// TODO: careful here
			&ptr[start],
			false, size_t(end - start),
			type, device
		};
	}

	// Copy from another resource
	bool copy(const Resource &r) {
		if (elements != r.elements)
			return false;

		// TODO: check that the device/API is the same
		if (ptr != r.ptr)
			std::memcpy(ptr, r.ptr, elements * sizeof(float));

		return true;
	}

	// Cloning resources
	std::optional <Resource> clone() const {
		// TODO: infer allocator
		float *new_ptr = nullptr;
		switch (device) {
		case eCPU:
			new_ptr = new float[elements];
		default:
			break;
		}

		if (ptr) {
			// TODO: depending on the device
			std::memcpy(new_ptr, ptr, elements * sizeof(float));
			return Resource { new_ptr, true, elements, type, device };
		}

		return std::nullopt;
	}

	// TODO: .to() function to transfer between devices

	static std::optional <Resource> from(size_t elements, Resource::Type type, Resource::Device device) {
		// TODO: custom allocator to track memory and hold pages for a particular device
		float *ptr = nullptr;
		switch (device) {
		case eCPU:
			ptr = new float[elements];
		default:
			break;
		}

		if (ptr)
			return Resource { ptr, true, elements, type, device };
		return std::nullopt;
	}

	static void drop(Resource &r) {
		if (r.owner)
			delete r.ptr;
		r.ptr = nullptr;
	}
};

// Printing utilities
std::string format_as(Resource::Type);
std::string format_as(Resource::Device);
