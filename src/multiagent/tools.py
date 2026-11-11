from langchain_core.tools import tool
from typing import Dict, Any

# Mock Orders Database
MOCK_ORDERS = {
    "ORD12345": {
        "status": "In Transit",
        "carrier": "FedEx Express",
        "tracking": "FX-9928172",
        "items": ["Wireless Noise-Canceling Headphones"],
        "estimated_delivery": "Tomorrow by 5:00 PM"
    },
    "ORD67890": {
        "status": "Delivered",
        "carrier": "UPS Ground",
        "tracking": "UPS-1102938",
        "items": ["Ergonomic Mechanical Keyboard"],
        "delivered_on": "Yesterday at 2:15 PM"
    },
    "ORD11223": {
        "status": "Processing",
        "carrier": "DHL Express",
        "tracking": "DHL-3392019",
        "items": ["4K USB-C Docking Station"],
        "estimated_delivery": "In 2 business days"
    },
}


@tool
def get_order_details(order_id: str) -> str:
    """Look up order status, shipping carrier, tracking number, and estimated delivery for an Order ID (e.g. ORD12345)."""
    clean_id = order_id.strip().upper()
    if clean_id in MOCK_ORDERS:
        info = MOCK_ORDERS[clean_id]
        delivery_info = info.get('estimated_delivery') or info.get('delivered_on', 'N/A')
        return f"Order {clean_id}: Status={info['status']}, Carrier={info['carrier']}, Tracking={info['tracking']}, Items={', '.join(info['items'])}, Delivery={delivery_info}"
    return f"Order ID '{order_id}' was not found in our database. Please verify your order number."


@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a refund for an order given the order_id, refund amount, and reason."""
    clean_id = order_id.strip().upper()
    if amount > 250.0:
        return f"Refund request of ${amount:.2f} for Order {clean_id} exceeds auto-approval limit ($250). Initiated manual manager review."
    return f"SUCCESS: Refund of ${amount:.2f} approved for Order {clean_id}. Reason: {reason}. Refund will post in 3-5 business days."


@tool
def escalate_complaint(issue_summary: str, customer_id: str = "CUST101") -> str:
    """Escalate a serious complaint or unresolved issue to Tier-2 Support Management."""
    return f"TICKET CREATED: Issue for customer '{customer_id}' escalated to Senior Management. Ticket ID: #ESC-{hash(issue_summary) % 10000:04d}. Priority: High."
