import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class OrderLookupResult:
    found: bool
    order_data: Optional[Dict[str, Any]]
    error: Optional[str]
    handoff_recommended: bool
    handoff_reason: Optional[str]

def normalize_order_id(order_id: str) -> str:
    """Normalize the order ID input.
    E.g., " ord-1007 " -> "ORD-1007", "ord1007" -> "ORD-1007", "1007" -> "ORD-1007".
    """
    if not order_id:
        return ""
    
    clean = order_id.strip().upper()
    # Match pattern ORD-XXXX or ORDXXXX or XXXX or ORD - XXXX
    match = re.match(r'^(?:ORD)?\s*[-_]?\s*(\d{4})$', clean)
    if match:
        return f"ORD-{match.group(1)}"
    
    return clean

def validate_order_id(order_id: str) -> bool:
    """Check if the normalized order ID is valid in format (must be ORD-XXXX)."""
    return bool(re.match(r'^ORD-\d{4}$', order_id))

def lookup_order(order_id: str, orders_file_path: str = None) -> OrderLookupResult:
    """Deterministic order lookup with sanitization and validation."""
    if not orders_file_path:
        # Default path relative to this workspace
        orders_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'orders.json'
        )

    # Normalize ID
    normalized_id = normalize_order_id(order_id)
    
    # Validate
    if not validate_order_id(normalized_id):
        return OrderLookupResult(
            found=False,
            order_data=None,
            error=f"Invalid order ID format: '{order_id}'. Order IDs must follow the format ORD-XXXX.",
            handoff_recommended=False,
            handoff_reason=None
        )

    # Load file
    if not os.path.exists(orders_file_path):
        return OrderLookupResult(
            found=False,
            order_data=None,
            error="Orders database is currently unavailable.",
            handoff_recommended=True,
            handoff_reason="Database file not found."
        )

    try:
        with open(orders_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return OrderLookupResult(
            found=False,
            order_data=None,
            error=f"Error loading orders database: {str(e)}",
            handoff_recommended=True,
            handoff_reason="Internal system error reading database."
        )

    # Find the order
    orders_list = data.get("orders", [])
    matched_order = None
    for o in orders_list:
        if o.get("order_id") == normalized_id:
            matched_order = o
            break

    if not matched_order:
        return OrderLookupResult(
            found=False,
            order_data=None,
            error=f"Order {normalized_id} was not found in our records.",
            handoff_recommended=True,
            handoff_reason="Order ID not found in database."
        )

    # Sanitize order data - extract customer-safe fields only
    status = matched_order.get("status")
    
    # 1. Base sanitized items (filter out SKU or any internal field)
    sanitized_items = []
    for item in matched_order.get("items", []):
        sanitized_items.append({
            "name": item.get("name"),
            "quantity": item.get("quantity"),
            "final_sale": item.get("final_sale", False)
        })

    sanitized_data = {
        "order_id": matched_order.get("order_id"),
        "membership_tier": matched_order.get("membership_tier"),
        "items": sanitized_items,
        "placed_at": matched_order.get("placed_at"),
        "status": status,
        "status_updated_at": matched_order.get("status_updated_at"),
        "customer_safe_message": matched_order.get("customer_safe_message")
    }

    # 2. Add shipping details depending on status rules
    # Exclude stale delivery estimates for cancelled or returned orders
    if status in ["cancelled", "returned"]:
        sanitized_data["estimated_delivery"] = None
        sanitized_data["carrier"] = None
        sanitized_data["tracking_number"] = None
        sanitized_data["shipped_at"] = None
        sanitized_data["delivered_at"] = None
    else:
        sanitized_data["estimated_delivery"] = matched_order.get("estimated_delivery")
        sanitized_data["carrier"] = matched_order.get("carrier")
        sanitized_data["tracking_number"] = matched_order.get("tracking_number")
        sanitized_data["shipped_at"] = matched_order.get("shipped_at")
        sanitized_data["delivered_at"] = matched_order.get("delivered_at")

    # 3. Handle special status codes
    handoff_recommended = False
    handoff_reason = None
    
    if status == "exception":
        handoff_recommended = True
        handoff_reason = "Order status is 'exception'. Requires human specialist investigation."

    return OrderLookupResult(
        found=True,
        order_data=sanitized_data,
        error=None,
        handoff_recommended=handoff_recommended,
        handoff_reason=handoff_reason
    )
