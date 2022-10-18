#include <cassert>
#include <chrono>
#include <deque>
#include <iostream>
#include <map>
#include <random>
#include <vector>
using namespace std;

typedef unsigned long long ull;

// priority queue base class for comparison purposes
template<class T>
class Q
{
public:
	virtual ~Q() = default;
	virtual void add_with_priority(size_t v, T w) = 0;
	virtual void decrease_priority(size_t v, T w) = 0;
	virtual size_t extract_min() = 0;
};

// simple priority queue 
template<class T = ull>
class PQ : public Q<T>
{
	typedef multimap<ull, size_t> M;
	M m;
	vector<M::iterator> its;
public:
	PQ(size_t n) :
		its(n)
	{

	}
	void add_with_priority(size_t v, T w) override
	{
		its[v] = m.emplace(w, v);
	}
	void decrease_priority(size_t v, T w) override
	{
		m.erase(its[v]);
		add_with_priority(v, w);
	}
	size_t extract_min() override
	{
		auto it = begin(m);
		auto ret = it->second;
		m.erase(it);
		return ret;
	}
};

#ifdef _DEBUG
#define DEBUG_ASSERT(x, ...) assert(x, ...)
#else
#define DEBUG_ASSERT(x)
#endif

// T : tipo guardado em cada vértice
template<class T = ull>
class FibHeap : public Q<T>
{
	struct Node
	{
		T w;
		size_t
			index,
			nChild;
		Node
			* pai, // if nullptr, it's a root node
			* child;
		// this union is just for readability
		union
		{
			struct // used by root node
			{
				Node
					* prv, // roots are circular doubly-linked list
					* nxt;
			};
			struct // used by non-root node
			{
				Node
					* prvBrother,
					* nxtBrother; // brothers are (not-circular) doubly-linked list
				bool lost; // if lost 1 child already (only used by non-root node)
			};
		};
	};
	vector<Node> nodes;
	vector<Node*>
		ranks, // auxiliar vec used when "consolidating" trees
		nodesPtr; // maps index to where it is currently

	size_t curRootUnused;
	Node* gMin, * firstUnused;

	void moveToARoot(Node* ptr)
	{
		if (ptr->prvBrother)
		{
			ptr->prvBrother->nxtBrother = ptr->nxtBrother;
			if (ptr->nxtBrother)
				ptr->nxtBrother->prvBrother = ptr->prvBrother;
		}
		else
		{
			ptr->pai->child = ptr->nxtBrother;
			if (ptr->pai->child)
				ptr->pai->child->prvBrother = nullptr;
		}
		gMin->nxt->prv = ptr;
		ptr->pai->nChild--;
		ptr->pai = nullptr;
		ptr->prv = gMin;
		ptr->nxt = exchange(gMin->nxt, ptr);
	}
	void link(Node* v, Node* r)
	{
		v->prv->nxt = v->nxt;
		v->nxt->prv = v->prv;
		if (r->child)
		{
			r->child->prvBrother = v;
			v->nxtBrother = r->child;
		}
		else
			v->nxtBrother = nullptr;
		r->child = v;
		r->nChild++;
		v->prvBrother = nullptr;
		v->pai = r;
		v->lost = false;
	}
public:
	// n : máximo número de vértices, cada um será indexado [0, n)
	FibHeap(size_t n) :
		nodes(n),
		ranks(n), // in theory, the rank is O(log(n)), so maybe you can optimize this (not worth it?)
		nodesPtr(n),
		curRootUnused(0),
		gMin(nullptr),
		firstUnused(nodes.data())
	{
		DEBUG_ASSERT(n); // n > 0

		// build singly-linked list for unused
		for (size_t i = 0, lim = n - 1; i != lim; i++)
			nodes[i].nxt = &nodes[i + 1];
	}
	void add_with_priority(size_t v, size_t w) override
	{
		DEBUG_ASSERT(v < nodes.size()); // adding a vertex outside of range

		nodesPtr[v] = firstUnused;
		auto& newRoot = *firstUnused;
		auto nodePtr = firstUnused;
		firstUnused = firstUnused->nxt;
		newRoot.index = v;
		newRoot.w = w;
		newRoot.pai = nullptr;
		newRoot.nChild = 0;
		if (gMin)
		{
			newRoot.nxt = gMin->nxt;
			gMin->nxt = nodePtr;
			newRoot.prv = gMin;
			newRoot.nxt->prv = nodePtr;
			if (w < gMin->w)
				gMin = nodePtr;
		}
		else
		{
			gMin = nodePtr;
			newRoot.prv = newRoot.nxt = nodePtr;
		}
	}
	// remove o menor vértice, e retorna seu índice (não chamar se a estrutura estiver vazia)
	size_t extract_min() override
	{
		DEBUG_ASSERT(gMin); // extract_min called on empty FibHeap

		auto ret = gMin->index;

		auto child = gMin->child;
		while (child)
		{
			auto brother = child->nxtBrother;
			moveToARoot(child);
			child = brother;
		}

		// delete min
		gMin->prv->nxt = gMin->nxt;
		gMin->nxt->prv = gMin->prv;
		auto itRoot = gMin->nxt; // pointer to any root
		auto willBeEmpty = itRoot == gMin;
		gMin->nxt = exchange(firstUnused, gMin);

		if (willBeEmpty)
			gMin = nullptr;
		else
		{
			// find new min and "consolidate" trees
			// (join same rank ones, where rank of a tree is
			// the number of direct children of the root)
			gMin = itRoot;
			ranks[gMin->nChild] = gMin;
			size_t maxRank = gMin->nChild;

			itRoot = itRoot->nxt;
			auto start = gMin;
			while (itRoot != start)
			{
				const auto& ourW = itRoot->w;
				if (ourW < gMin->w)
					gMin = itRoot;
				const auto& curNChild = itRoot->nChild;
				if (curNChild > maxRank)
					maxRank = curNChild;
				auto nxt = itRoot->nxt;
				if (auto& cur = ranks[curNChild])
				{
					auto& curW = cur->w;
					if (ourW < curW)
					{
						if (start == cur)
							start = start->nxt;
						link(cur, itRoot);
						nxt = itRoot->nxt; // caso start == cur e start->nxt fosse itRoot (caso com 2 árvores)
						cur = itRoot;
					}
					else
					{
						nxt = itRoot->nxt;
						link(itRoot, cur);
					}
				}
				else
					cur = itRoot;
				itRoot = nxt;
			}
			// clear ranks vec for a future iteration
			for (size_t i = 0; i <= maxRank; i++)
				ranks[i] = nullptr;
		}

		return ret;
	}
	// atualiza o peso de v para w
	void decrease_priority(size_t v, size_t w) override
	{
		const auto& nodePtr = nodesPtr[v];
		auto& node = *nodePtr;
		DEBUG_ASSERT(w <= node.w);
		node.w = w;
		if (auto paiPtr = node.pai)
		{
			if (w < paiPtr->w)
			{
				moveToARoot(nodePtr);
				while (auto nxt = paiPtr->pai) // enquanto não é root, vai "podando" (criando novas árvores caso já tivesse perdido um filho)
				{
					auto& lost = paiPtr->lost;
					if (lost)
					{
						moveToARoot(paiPtr);
						paiPtr = nxt;
					}
					else
					{
						lost = true;
						break;
					}
				}
			}
		}
		if (w < gMin->w)
			gMin = nodePtr;
	}
};

template<class T = ull>
struct G
{
	typedef deque<pair<size_t, T>> N;
	vector<N> adj;
	G(size_t n) :
		adj(n)
	{

	}
	void add(size_t i, size_t j, T w)
	{
		adj[i].emplace_back(j, w);
	}
	size_t size() const
	{
		return adj.size();
	}
	const N& operator[](size_t v) const
	{
		return adj[v];
	}
};

template<template<class>class PriorityQueue, class T>
auto Dijkstra(const G<T>& g)
{
	static constexpr auto INF = numeric_limits<T>::max();
	static constexpr auto UNDEF = INF;

	const auto n = g.size();
	vector<T> dist(n);
	vector<size_t> prv(n);
	PriorityQueue Q(n);
	for (size_t v = 0; v != n; v++)
	{
		if (v)
		{
			dist[v] = INF;
			prv[v] = UNDEF;
		}
		Q.add_with_priority(v, dist[v]);
	}
	for (size_t i = 0; i != n; i++)
	{
		auto u = Q.extract_min(); // remove and return best vertex
		if (dist[u] != INF) // (handle overflow)
			for (const auto& [v, w] : g[u])
			{
				const auto alt = dist[u] + w;
				if (alt < dist[v])
				{
					dist[v] = alt;
					prv[v] = u;
					Q.decrease_priority(v, alt);
				}
			}
	}
	return make_pair(dist, prv);
}

constexpr unsigned SEED = 24;
mt19937 mt(SEED);

size_t getRandomSizeT(size_t a, size_t b)
{
	uniform_int_distribution<size_t> d(a, b);
	return d(mt);
}

double getRandomDouble(double a, double b)
{
	uniform_real_distribution d(a, b);
	return d(mt);
}

// cria um grafo com n vértices, probabilidade p de existir cada aresta, faz a média de T execuções
pair<double, double> TestNP(size_t n, double p, ull minW=0, ull maxW=1000000, size_t T=2)
{
	double a = 0, b = 0;
	for (size_t i = 0; i != T; i++)
	{
		G g(n);
		for (size_t i = 0; i != n; i++)
			for (size_t j = 0; j != n; j++)
				if (mt() & 1)
					g.add(i, j, getRandomSizeT(minW, maxW));

		auto t1 = chrono::high_resolution_clock::now();
		auto best1 = Dijkstra<PQ>(g);
		auto t2 = chrono::high_resolution_clock::now();
		auto ms_int1 = duration_cast<chrono::milliseconds>(t2 - t1).count();

		auto t3 = chrono::high_resolution_clock::now();
		auto best2 = Dijkstra<FibHeap>(g);
		auto t4 = chrono::high_resolution_clock::now();
		auto ms_int2 = duration_cast<chrono::milliseconds>(t4 - t3).count();

		assert(best1.first == best2.first);

		a += ms_int1;
		b += ms_int2;
	}
	return { a / T, b / T };
}

int main()
{
	// sparse graphs (10% edges)
	// (50% edges)
	// dense graphs (90% edges)
	static constexpr pair<double, size_t> TESTS[]{
		{.1, 100},
		{.5, 100},
		{.9, 100},
		{.1, 1000},
		{.5, 1000},
		{.9, 1000},
		{.1, 40000},
		{.5, 40000},
		{.9, 20000},
		{.1, 20000},
		{.6, 20000}
	};
	for (const auto& [p, n] : TESTS)
		{
			auto [a, b] = TestNP(n, p);
			cout << "n=" << n << ", p=" << p << ": " << a << '\t' << b << endl;
		}
}