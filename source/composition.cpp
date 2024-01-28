#include "composition.hpp"

// Converting proxy chains
ChainProxy::operator Chain()
{
	std::vector <Function *> transfered(begin(), end());
	return Chain::from(transfered);
}
