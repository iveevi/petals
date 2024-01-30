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
	constexpr size_t TRAIN_BATCHES = 600;
	constexpr size_t BATCH_SIZE = 100;
	constexpr size_t VALIDATION_SIZE = 1000;
	constexpr size_t EPOCHS = 10;

	static_assert(TRAIN_BATCHES * BATCH_SIZE <= 60000);

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
	// TODO: single tensor here
	std::vector <Tensor> tXs;
	std::vector <Tensor> tYs;

	for (size_t n = 0; n < TRAIN_BATCHES; n++) {
		Tensor tX = Tensor::zeros({ BATCH_SIZE, IMAGE_SIZE });
		Tensor tY = Tensor::zeros({ BATCH_SIZE, 10ul });

		for (size_t i = 0; i < BATCH_SIZE; i++) {
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

		tXs.push_back(tX);
		tYs.push_back(tY);
	}

	f_train_images.close();
	f_train_labels.close();

	// Loading validation data
	Tensor vX = Tensor::zeros({ VALIDATION_SIZE, IMAGE_SIZE });
	Tensor vY = Tensor::zeros({ VALIDATION_SIZE, 10ul });

	for (size_t i = 0; i < VALIDATION_SIZE; i++) {
		// Read the image
		unsigned char image[IMAGE_SIZE + 1] { 0 };
		f_validation_images.read((char *) image, IMAGE_SIZE);

		for (size_t j = 0; j < IMAGE_SIZE; j++) {
			size_t index = i * IMAGE_SIZE + j;
			vX.buffer.ptr[index] = image[j] / 255.0f;
		}

		// Read the label
		unsigned char label;
		f_validation_labels.read((char *) &label, 1);

		for (size_t j = 0; j < 10; j++) {
			size_t index = i * 10 + j;
			vY.buffer.ptr[index] = (label == j);
		}
	}

	f_validation_images.close();
	f_validation_labels.close();

	// Construct the model
	// Chain model = Linear::from(IMAGE_SIZE, 30) >> Linear::from(30, 10) >> ops::softmax;
	Chain model = Linear::from(IMAGE_SIZE, 30) >> ops::sigmoid >> Linear::from(30, 10) >> ops::softmax;

	auto validation_score = [&]() {
		Tensor pY = model(vX);

		size_t correct = 0;
		for (size_t i = 0; i < VALIDATION_SIZE; i++) {
			size_t max_predicted = 0;
			float max_predicted_value = 0.0f;
			for (size_t j = 0; j < 10; j++) {
				size_t index = i * 10 + j;
				float x = pY.buffer.ptr[index];
				if (x > max_predicted_value) {
					max_predicted_value = x;
					max_predicted = j;
				}
			}

			size_t max_true = 0;
			float max_true_value = 0.0f;
			for (size_t j = 0; j < 10; j++) {
				size_t index = i * 10 + j;
				float x = vY.buffer.ptr[index];
				if (x > max_true_value) {
					max_true_value = x;
					max_true = j;
				}
			}

			// if (i == 0) {
			// 	fmt::print("pY = {}\namax = {} -> avalue = {} -> label = {}\n", pY[i], max_predicted, max_predicted_value, max_true);
			// }

			correct += (max_predicted == max_true);
		}

		return correct/float(VALIDATION_SIZE);
	};

	// auto opt = Momentum::from(model.parameters(), 0.001f);
	auto opt = Adam::from(model.parameters(), 0.01f);

	for (size_t n = 0; n < EPOCHS; n++) {
		fmt::print("\n\nepoch {}, accuracy {}\n", n, validation_score());
		for (size_t i = 0; i < TRAIN_BATCHES; i++) {
			// TODO: we can just slice...
			const Tensor &tX = tXs[i];
			const Tensor &tY = tYs[i];

			// fmt::print("layer: {}\n", (*model.parameters()[1])[0]);
			auto predicted = model(tX);
			// fmt::print("predicted: {}\n", predicted);
			// fmt::print("loss: {}\n", sum(square(predicted - tY)));

			Tape tape = Tape::from(model.parameters());

			// TODO: mean
			auto loss = sum(square(predicted - tY))/10;
			// fmt::print("loss graph: {}\n", loss);
			// fmt::print("  > loss: {}\n", loss.eval());
			loss.eval();
			loss.backward(tape);

			// fmt::print("tape: {}\n", tape.size());
			// for (const auto &[tag, grad] : tape)
			// 	fmt::print("  {} -> {}\n", tag, sum(grad).eval().buffer.ptr[0]);

			opt.step(tape);
		}
	}
}
