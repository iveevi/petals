#include <filesystem>

#include "ops.hpp"
#include "composition.hpp"

static const std::filesystem::path DATA_DIRECTORY = "data";
static const std::unordered_map <std::string, std::string> DATA_URLS {
	{ "train-images-idx3-ubyte", "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" },
	{ "train-labels-idx1-ubyte", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" },
	{ "t10k-images-idx3-ubyte",  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"  },
	{ "t10k-labels-idx1-ubyte",  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"  }
};

int main()
{
	constexpr size_t IMAGE_SIZE = 784;
	// TODO: pass config to this or choose the config

	// Create the data directory
	std::filesystem::create_directory(DATA_DIRECTORY);
	for (const auto &[file, url] : DATA_URLS) {
		if (!std::filesystem::exists(DATA_DIRECTORY / file)) {
			system(("wget " + url + " -P " + DATA_DIRECTORY.string()).c_str());
			system(("gunzip " + (DATA_DIRECTORY / file).string() + ".gz").c_str());
		}
	}

	// Construct the model
	Chain model = Linear::from(IMAGE_SIZE, 30) >> ops::sigmoid >> Linear::from(30, 10);
}
