#pragma once

#include <cstring>
#include <optional>
#include <string>
#include <atomic>

#include <fmt/core.h>

struct Resource {
	// TODO: variant of all pointer types and vk buffer
	double *ptr;
	size_t elements;

	// Tracking
	std::atomic <long long int> *counter;

	enum Type {
		f32
	} type;

	enum Device {
		eCPU,
		eCUDA,
		eVulkan
	} device;

	// Enabling debug tracking
	bool tracking = false;

	// Initializer list
	Resource(double *_ptr = nullptr, size_t _elements = 0, std::atomic <long long int> *_counter = nullptr, Type _type = Type::f32, Device _device = Device::eCPU)
			: ptr(_ptr), elements(_elements), counter(_counter), type(_type), device(_device) {
		if (tracking)
			fmt::print("delegating resource, counter = {}/{}\n", (void *) counter, counter ? counter->load() : -1);
	}

	// Copy constructor(s)
	Resource(const Resource &other) {
		// TODO: delegate constructor
		ptr = other.ptr;
		elements = other.elements;
		counter = other.counter;
		type = other.type;
		device = other.device;
		tracking = other.tracking;

		// Add use
		if (counter)
			(*counter)++;

		// TODO: mode to profile/count number of copy transactions
		if (other.tracking)
			fmt::print("copy resource (Constructor), counter = {}/{}\n", (void *) counter, counter ? counter->load() : -1);
	}

	Resource &operator=(const Resource &other) {
		if (this == &other)
			return *this;

		// If there was a valid counter before, lower that (no longer beind used)
		drop();

		// TODO: delegate constructor
		// TODO: combine into a function to avoid future bugs
		ptr = other.ptr;
		elements = other.elements;
		counter = other.counter;
		type = other.type;
		device = other.device;
		tracking = other.tracking;

		// Add use
		if (counter)
			(*counter)++;

		// TODO: mode to profile/count number of copy transactions
		if (other.tracking)
			fmt::print("copy resource (Operator=), counter = {}/{}\n", (void *) counter, counter ? counter->load() : -1);

		return *this;
	}

	~Resource() {
		drop();
	}

	// Memset each element
	void memset(double value) const {
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
			size_t(end - start),
			nullptr,
			type, device
		};
	}

	// Copy from another resource
	bool copy(const Resource &r) {
		if (elements != r.elements)
			return false;

		// TODO: check that the device/API is the same
		if (ptr != r.ptr)
			std::memcpy(ptr, r.ptr, elements * sizeof(double));

		return true;
	}

	// Cloning resources
	std::optional <Resource> clone() const {
		// TODO: infer allocator
		double *new_ptr = nullptr;
		switch (device) {
		case eCPU:
			new_ptr = new double[elements] { 0 };
		default:
			break;
		}

		if (ptr) {
			// TODO: depending on the device
			std::atomic <long long int> *new_counter = new std::atomic <long long int> (1);
			std::memcpy(new_ptr, ptr, elements * sizeof(double));
			if (tracking)
				fmt::print("[!!] CLONED RESOURCE: {} elements @{}\n", elements, (void *) new_ptr);
			return Resource { new_ptr, elements, new_counter, type, device };
		}

		return std::nullopt;
	}

	// TODO: .to() function to transfer between devices

	static std::optional <Resource> from(size_t elements, Resource::Type type, Resource::Device device) {
		// TODO: custom allocator to track memory and hold pages for a particular device
		double *ptr = nullptr;
		switch (device) {
		case eCPU:
			ptr = new double[elements] { 0 };
		default:
			break;
		}

		if (ptr) {
			// fmt::print("[!!] NEW RESOURCE: {} elements @{}\n", elements, (void *) ptr);
			std::atomic <long long int> *counter = new std::atomic <long long int> (1);
			return Resource { ptr, elements, counter, type, device };
		}

		return std::nullopt;
	}
private:
	// Manually dropping count and optional deallocation
	void drop() {
		// Only attempt free if it is the original source
		if (counter) {
			(*counter)--;
			if (tracking)
				fmt::print("destructor for original source @{} --> {}/{}\n", (void *) ptr, (void *) counter, counter ? counter->load() : -1);
			if (counter->load() == 0) {
				if (tracking)
					fmt::print("--> DESTROYING RESOURCE @{}\n", (void*) ptr);
				delete counter;
				delete[] ptr;
			}
		}

		counter = nullptr;
		ptr = nullptr;
	}
};

// Printing utilities
std::string format_as(Resource::Type);
std::string format_as(Resource::Device);
