#pragma once
#include "market_data.h"
#include <cstdint>

namespace market {

/**
 * @brief Zero-copy message parser with validation and sequence checking
 */
class MessageParser {
public:
    /**
     * @brief Parse a raw message from the network layer
     * @param raw The raw message from UDP receiver (zero-copy)
     * @return Pointer to validated MessageHeader, or nullptr if invalid
     */
    const MessageHeader* parse(const RawMessage& raw);

    /**
     * @brief Type-safe casting to specific message types
     * @tparam T The message type to cast to (must inherit from MessageHeader)
     * @param header Pointer to a validated MessageHeader
     * @return Typed pointer to the message structure
     */
    template <typename T>
    const T* as(const MessageHeader* header) const {
        return reinterpret_cast<const T*>(header);
    }

    /**
     * @brief Get total sequence gaps detected
     */
    uint64_t sequence_gaps() const;

    /**
     * @brief Get count of invalid messages rejected
     */
    uint64_t invalid_messages() const;

private:
    /**
     * @brief Compute expected message length for a given type
     * @param header The message header to check
     * @return Expected length in bytes, or 0 for invalid types
     */
    uint64_t compute_expected_len(const MessageHeader* header) const;

    uint32_t last_sequence_{0};    ///< Last valid sequence number received
    uint64_t gaps_{0};            ///< Total sequence gaps detected
    uint64_t invalid_{0};         ///< Total invalid messages rejected
};

}  

} 

