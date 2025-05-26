#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./include/MurmurHash3.h"

using namespace std;

class TokenGen {
   public:
    TokenGen(const string_view sv, const string_view del) : sv_(sv), del_(del) {}

    operator bool() const { return !sv_.empty(); }

    string_view operator()() {
        while (true) {
            auto r = sv_;
            const auto it = sv_.find_first_of(del_);
            if (it == string_view::npos) {
                sv_ = {};
            } else {
                r.remove_suffix(r.size() - it);
                sv_.remove_prefix(it + 1);
            }
            if (!r.empty()) {
                return r;
            }
            if (sv_.empty()) {
                return {};
            }
        }
    }

   private:
    string_view sv_;
    string_view del_;
};

// Implemented based on https://github.com/apache/spark/blob/82e3f0d5d594f544ec4689cb833879c8a95ec849/mllib/src/main/scala/org/apache/spark/ml/feature/MinHashLSH.scala#L163
class MinHasher {
   public:
    MinHasher(size_t num_hashes, uint32_t seed = 1) : num_hashes_(num_hashes) {
        mt19937 rng_(seed);
        uniform_int_distribution<uint32_t> dist_a_(1, kHashPrime - 1);
        uniform_int_distribution<uint32_t> dist_b_(0, kHashPrime - 1);

        a_.resize(num_hashes_);
        b_.resize(num_hashes_);
        for (size_t i = 0; i < num_hashes_; ++i) {
            a_[i] = dist_a_(rng_);
            b_[i] = dist_b_(rng_);
        }
    }

    // Compute MinHash signature for a set of feature indices
    vector<uint32_t> compute_signature(const unordered_set<size_t>& feature_indices) const {
        vector<uint32_t> sig(num_hashes_, kHashPrime);
        for (const auto& idx : feature_indices) {
            for (size_t i = 0; i < num_hashes_; ++i) {
                uint32_t hash = static_cast<uint32_t>(
                    ((1ULL + idx) * a_[i] + b_[i]) % kHashPrime);
                sig[i] = min(sig[i], hash);
            }
        }
        return sig;
    }

    // Jaccard distance between two MinHash signatures
    static double jaccard_distance(const vector<uint32_t>& a, const vector<uint32_t>& b) {
        size_t match = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] == b[i]) {
                match++;
            }
        }
        return 1.0 - static_cast<double>(match) / a.size();
    }

   private:
    static constexpr uint32_t kHashPrime = 2038074743;

    size_t num_hashes_;
    vector<uint32_t> a_;
    vector<uint32_t> b_;
};

class Deduplicator {
   public:
    Deduplicator(size_t ngrams, size_t num_hashes, double threshold, size_t num_features) : ngrams_(ngrams),
                                                                                            num_features_(num_features),
                                                                                            threshold_(threshold),
                                                                                            hasher_(num_hashes) {
        for (int c = 0; c <= numeric_limits<unsigned char>::max(); ++c) {
            if (!isalnum(c)) {
                nonalnum_ += static_cast<char>(c);
            }
        }
    }

    void process(const vector<string_view>& docs) {
        vector<vector<uint32_t>> signatures;
        signatures.reserve(docs.size());
        for (const string_view doc : docs) {
            const auto& ngrams = extract_features(doc);
            signatures.emplace_back(hasher_.compute_signature(ngrams));
        }

        for (size_t i = 0; i < docs.size(); ++i) {
            for (size_t j = i + 1; j < docs.size(); ++j) {
                const auto dist = MinHasher::jaccard_distance(signatures[i], signatures[j]);
                if (dist < threshold_) {
                    cout << "Duplicate pair (Jaccard: " << 1.0 - dist << "):\n"
                         << " - " << docs[i] << "\n - " << docs[j] << "\n\n";
                }
            }
        }
    }

   private:
    size_t ngrams_;
    size_t num_features_;
    double threshold_;
    string nonalnum_;
    MinHasher hasher_;

    unordered_set<size_t> extract_features(const string_view text) {
        deque<string_view> window;
        unordered_set<size_t> indices;

        TokenGen splitter(text, nonalnum_);
        while (splitter) {
            const auto token = splitter();
            window.emplace_back(token);

            if (window.size() > ngrams_) {
                window.pop_front();
            }
            if (window.size() == ngrams_) {
                string combined;
                for (const auto& w : window) {
                    combined += w;
                    combined += "_";
                }

                uint32_t h{};
                MurmurHash3_x86_32(combined.data(), combined.size(), 0, &h);
                indices.insert(h % num_features_);
            }
        }

        return indices;
    }
};

int main() {
    const vector<string_view> data = {
        "Since 2000, the Vatreni have qualified for every major tournament except UEFA Euro 2000 and the 2010 FIFA World Cup. At the World Cup, Croatia has finished second once (2018) and third on two occasions (1998, 2022), securing three World Cup medals. Davor Šuker won the Golden Shoe and the Silver Ball in 1998, while Luka Modrić won the Golden Ball in 2018 and the Bronze Ball in 2022. The team has reached the quarter-finals of the UEFA European Championship twice (1996, 2008). They finished second in the UEFA Nations League in 2023.",
        "Since 2000, the Vatreni have not qualified for every minor tournament except for the 2010 FIFA World Cup. At the World Cup, Croatia has finished second once (2018) and third on two occasions (1998, 2022), securing three World Cup medals. Davor Šuker won the Golden Shoe and the Silver Ball in 1998, while Luka Modrić won the Golden Ball in 2018 and the Bronze Ball in 2022. The team has not reached the quarter-finals of the UEFA European Championship twice (1996, 2008). They finished third in the UEFA Nations League in 2023.",
        "Roses are red, my love, doo-roo-roo-roo\nA long-long time ago, on graduation day\nYou handed me your book, I signed this way\nRoses are red, my love, violets are blue\nSugar is sweet, my love, but not as sweet as you\nWe dated through high school, and when the big day came\nI wrote into your book, next to my name\nRoses are red, my love, violets are blue\nSugar is sweet, my love, but not as sweet as you\n(As sweet as you)\nThen I went far away and you found someone new\nI read your letter, dear, and I wrote back to you\nRoses are red, my love, violets are blue\nSugar is sweet, my love, good luck, may God bless you\n(May God bless you)\nIs that your little girl? She looks a lot like you\nSomeday, some boy will write in her book too\nRoses are red, my love, violets are blue\nSugar is sweet, my love, but not as sweet as you\nRoses are red\n",
        "Roses are blue, my love, doo-roo-roo-roo\nA long-long time ago, on graduation day\nYou handed me your book, I signed this way\nRoses are blue, my love, violets are blue\nSugar is sweet, my love, but not as sweet as you\nWe dated through high school, and when the big day came\nI wrote into your book, next to my name\nRoses are blue, my love, violets are blue\nSugar is sweet, my love, but not as sweet as you\n(As sweet as you)\nThen I went far away and you found someone new\nI read your letter, dear, and I wrote back to you\nRoses are blue, my love, violets are blue\nSugar is sweet, my love, good luck, may God bless you\n(May God bless you)\nIs that your little girl? She looks a lot like you\nSomeday, some boy will write in her book too\nRoses are blue, my love, violets are blue\nSugar is sweet, my love, but not as sweet as you\nRoses are blue\n",
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "different than the others",
        "As described in RFC 2606 and RFC 6761, a number of domains such as example.com and example.org are maintained for documentation purposes. These domains may be used as illustrative examples in documents without prior coordination with us. They are not available for registration or transfer. We provide a web service on the example domain hosts to provide basic information on the purpose of the domain. These web services are provided as best effort, but are not designed to support production applications. While incidental traffic for incorrectly configured applications is expected, please do not design applications that require the example domains to have operating HTTP service.",
        "As described in RFC 11111 or RFC 6761, many domains such as example.com and example.org are maintained for various purposes. These domains may be used as illustrative examples in documents without previously coordinating with us. They are not available for registration or transfer. We provide a web service on the example domain hosts to provide basic information on the purpose of the domain. These web services are provided as best effort, but are not designed to support production applications. While incidental traffic for misconfigured applications is expected, please do not design applications that require the example domains to have operating HTTP service.",
        // "",
        // "|||",
        // "||| a",
        // "a|||",
        // "AAAA,,,,,,|| ||",
    };

    // 262144 is default in https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.HashingTF.html
    // Should be a power of 2
    Deduplicator dedup(3, 13, 0.3, 262144);
    dedup.process(data);

    return 0;
}
