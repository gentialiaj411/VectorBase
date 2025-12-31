#pragma once
#include "market_data.h"
#include <cstdint>
#include <iostream>
#include <map>
#include <unordered_map>

namespace market {

struct Order {
    uint64_t order_id{};    ///< Unique order identifier
    uint32_t symbol_id{};   ///< Which symbol this order is for
    int64_t price{};        ///< Order price level (fixed-point)
    uint32_t size{};        ///< Order quantity
    char side{0};           ///< 'B' for buy, 'S' for sell
};

class OrderBook {
public:
    /**
     * @brief Add a new order to the book
     */
    void on_order_add(const OrderAdd& msg);

    /**
     * @brief Cancel an existing order=
     */
    void on_order_cancel(const OrderCancel& msg);

    /**
     * @brief Update best bid/ask prices (quote update)
     */
    void on_quote(const Quote& msg);

    /**
     * @brief Get the best (highest) bid price
     */
    int64_t best_bid() const;

    /**
     * @brief Get the best (lowest) ask price
     */
    int64_t best_ask() const;

    /**
     * @brief Calculate bid-ask spread
     * @return Spread in price units, or 0 if market not available
     */
    int64_t spread() const;

    /**
     * @brief Display top N price levels on both sides
     * @param n Number of levels to display (default 5)
     */
    void print_top_levels(int n = 5) const;

private:
    /**
     * Map key = price, value = total size at that price level
     */
    std::map<int64_t, uint32_t, std::greater<>> bids_;

    /**
     * Map key = price, value = total size at that price level
     */
    std::map<int64_t, uint32_t> asks_;

    /**
     * Order lookup table for cancellations
     */
    std::unordered_map<uint64_t, Order> orders_;
};

} 

} 

