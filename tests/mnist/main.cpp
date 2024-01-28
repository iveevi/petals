#include "tensor.hpp"
#include "autograd.hpp"

static const std::string DATA_DIRECTORY = "data";
static const std::unordered_map <std::string, std::string> DATA_URLS {
	{ "train-images-idx3-ubyte", "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" },
	{ "train-labels-idx1-ubyte", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" },
	{ "t10k-images-idx3-ubyte",  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"  },
	{ "t10k-labels-idx1-ubyte",  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"  }
};

int main()
{

}
