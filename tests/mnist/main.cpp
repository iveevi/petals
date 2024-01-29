#include <cassert>
#include <filesystem>
#include <fstream>

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
	constexpr size_t TRAIN_SIZE = 10;
	constexpr size_t EPOCHS = 5;

	// TODO: pass config to this or choose the config

	// Create the data directory
	std::filesystem::create_directory(DATA_DIRECTORY);
	for (const auto &[file, url] : DATA_URLS) {
		if (!std::filesystem::exists(DATA_DIRECTORY / file)) {
			system(("wget " + url + " -P " + DATA_DIRECTORY.string()).c_str());
			system(("gunzip " + (DATA_DIRECTORY / file).string() + ".gz").c_str());
		}
	}

	// Load the data
	// TODO: method that returns this...
	std::ifstream f_train_images(DATA_DIRECTORY / "train-images-idx3-ubyte");
	std::ifstream f_train_labels(DATA_DIRECTORY / "train-labels-idx1-ubyte");

	assert(f_train_images.good());
	assert(f_train_labels.good());

	std::ifstream f_validation_images(DATA_DIRECTORY / "t10k-images-idx3-ubyte");
	std::ifstream f_validation_labels(DATA_DIRECTORY / "t10k-labels-idx1-ubyte");

	assert(f_validation_images.good());
	assert(f_validation_labels.good());

	// Headers
	char header[16];

	f_train_images.read(header, 16);
	f_validation_images.read(header, 16);

	f_train_labels.read(header, 8);
	f_validation_labels.read(header, 8);

	// TODO: displaying these images
	Tensor tX = Tensor::zeros({ TRAIN_SIZE, IMAGE_SIZE });
	Tensor tY = Tensor::zeros({ TRAIN_SIZE, 10ul });

	for (size_t i = 0; i < TRAIN_SIZE; i++) {
		// Read the image
		unsigned char image[IMAGE_SIZE + 1] { 0 };
		f_train_images.read((char *) image, IMAGE_SIZE);

		for (size_t j = 0; j < IMAGE_SIZE; j++) {
			size_t index = i * IMAGE_SIZE + j;
			tX.buffer.ptr[index] = image[j] / 255.0f;
		}

		// Read the label
		unsigned char label;
		f_train_labels.read((char *) &label, 1);

		for (size_t j = 0; j < 10; j++) {
			size_t index = i * 10 + j;
			tY.buffer.ptr[index] = (label == j);
		}
	}

	f_train_images.close();
	f_train_labels.close();

	// Construct the model
	// Chain model = Linear::from(IMAGE_SIZE, 30) >> ops::sigmoid >> Linear::from(30, 10);
	// TODO: how to fix exploding or imploding values into softmax?
	Chain model = Linear::from(IMAGE_SIZE, 30) >> ops::sigmoid >> Linear::from(30, 10) >> ops::softmax;
	// Chain model = Linear::from(IMAGE_SIZE, 30) >> ops::sigmoid >> Linear::from(30, 10) >> ops::softmax;

	auto opt = SGD::from(model.parameters(), 1.0f);

	for (size_t i = 0; i < EPOCHS; i++) {
		fmt::print("layer: {}\n", (*model.parameters()[1])[0]);
		auto predicted = model(tX);
		// fmt::print("predicted: {}\n", predicted);
		// fmt::print("loss: {}\n", sum(square(predicted - tY)));

		Tape tape = Tape::from(model.parameters());

		// TODO: mean
		auto loss = sum(square(predicted - tY))/(TRAIN_SIZE);
		loss.eval();
		// fmt::print("loss graph: {}\n", loss);
		fmt::print("\nloss: {}\n", loss.eval());
		loss.backward(tape);

		fmt::print("tape: {}\n", tape.size());
		for (const auto &[tag, grad] : tape)
			fmt::print("  {} -> {}\n", tag, sum(grad).eval());

		opt.step(tape);
	}
}
