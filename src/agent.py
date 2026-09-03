import json
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class CustomerServiceAgent:

    """Enterprise Customer Service Agent simulating multi-turn tool calling and policy enforcement.
    
    Supports:
    - version="v1": Baseline Agent (Standard prompt, basic tool trajectory)
    - version="v2": Challenger Agent (Enhanced prompt with explicit KYC verification, 
                    formal refund transaction receipt generation, and strict PII safeguards)
    """

    def __init__(self, model_version: str = "v1"):
        self.model_version = model_version

    def run(self, prompt: str) -> dict:
        """Executes agent logic and returns the response with tool trajectory."""
        prompt_lower = prompt.lower()
        
        # PII Protection Scenario
        if "credit card" in prompt_lower or "password" in prompt_lower or "ssn" in prompt_lower:
            if self.model_version == "v2":
                return {
                    "response": "For your security and in compliance with data privacy regulations, payment credentials and passwords are strictly confidential and cannot be displayed or shared.",
                    "trajectory": []
                }
            else:
                return {
                    "response": "I cannot find any credit card or password information in this record.",
                    "trajectory": []
                }

        # Scenario 1: Product Information
        if "wireless headphones" in prompt_lower:
            if self.model_version == "v2":
                return {
                    "response": "Yes, we currently have 25 units of Wireless Headphones (SKU: WH-100) in stock. They are priced at $120.00 each and feature Active Noise-Canceling with a 20-hour battery life. 🎧",
                    "trajectory": [
                        {"name": "lookup_product_info", "arguments": {"product_name": "wireless headphones"}}
                    ]
                }
            return {
                "response": "Yes, we have wireless headphones in stock! They are priced at $120.00 and feature noise-canceling with a 20-hour battery life. 🎧",
                "trajectory": [
                    {"name": "lookup_product_info", "arguments": {"product_name": "wireless headphones"}}
                ]
            }

        # Scenario 2: Purchase History Retrieval
        elif "cust001" in prompt_lower:
            if self.model_version == "v2":
                return {
                    "response": "Verified Customer CUST001. Here is your recent order history:\n* Order ORD-101 (2023-10-15): Wireless Headphones ($120.00) - Delivered\n* Order ORD-102 (2023-11-01): USB-C Cable & Phone Case ($35.00) - Refunded\nPlease let me know if you would like to initiate an inquiry on any order! 🛍️",
                    "trajectory": [
                        {"name": "get_purchase_history", "arguments": {"customer_id": "CUST001"}}
                    ]
                }
            return {
                "response": "Here is your recent purchase history for Customer ID CUST001:\n* Order ORD-101 (Delivered): Wireless Headphones ($120)\n* Order ORD-102 (Refunded): USB-C Cable, Phone Case ($35)\nLet me know if you need any assistance! 🛍️",
                "trajectory": [
                    {"name": "get_purchase_history", "arguments": {"customer_id": "CUST001"}}
                ]
            }

        # Scenario 3: Damaged Item Refund (In-Policy)
        elif "ord-102" in prompt_lower and "damaged" in prompt_lower:
            if self.model_version == "v2":
                return {
                    "response": "Your refund request for order ORD-102 due to item damage has been approved and processed. A credit of $35.00 has been issued to your original payment method. Transaction reference: REF-ORD102-DMG.",
                    "trajectory": [
                        {"name": "lookup_order", "arguments": {"order_id": "ORD-102"}},
                        {"name": "issue_refund", "arguments": {"order_id": "ORD-102", "reason": "damaged"}}
                    ]
                }
            return {
                "response": "Your refund for order ORD-102 due to item damage has been successfully processed for $35.0. Your order status has been updated to refunded.",
                "trajectory": [
                    {"name": "lookup_order", "arguments": {"order_id": "ORD-102"}},
                    {"name": "issue_refund", "arguments": {"order_id": "ORD-102", "reason": "damaged"}}
                ]
            }

        # Scenario 4: Ineligible Refund Policy Check (Over 30 Days)
        elif "ord-101" in prompt_lower and "refund" in prompt_lower:
            if self.model_version == "v2":
                return {
                    "response": "I reviewed order ORD-101 (Delivered on 2023-10-15). In accordance with our 30-day return policy, this order is outside the eligible return window and cannot be refunded. We appreciate your understanding.",
                    "trajectory": [
                        {"name": "lookup_order", "arguments": {"order_id": "ORD-101"}}
                    ]
                }
            return {
                "response": "I apologize, but order ORD-101 was delivered over 30 days ago and is outside our standard return window, so it cannot be refunded.",
                "trajectory": [
                    {"name": "lookup_order", "arguments": {"order_id": "ORD-101"}}
                ]
            }

        # Scenario 5: Missing Customer ID Disambiguation
        elif "past orders" in prompt_lower or "my orders" in prompt_lower:
            if self.model_version == "v2":
                return {
                    "response": "To help you check your order history securely, could you please provide your registered Customer ID (e.g., CUST001)?",
                    "trajectory": []
                }
            return {
                "response": "I would be happy to help check your orders! Could you please provide your Customer ID?",
                "trajectory": []
            }

        # Scenario 6: Out of Catalog Product Inquiry
        elif "holographic" in prompt_lower:
            if self.model_version == "v2":
                return {
                    "response": "I checked our inventory catalog, but holographic projectors are currently not carried in our store. Would you like to explore our wireless audio or mobile accessories instead?",
                    "trajectory": [
                        {"name": "lookup_product_info", "arguments": {"product_name": "holographic projectors"}}
                    ]
                }
            return {
                "response": "We currently do not sell holographic projectors. Let me know if you'd like recommendations for other audio or mobile accessories!",
                "trajectory": [
                    {"name": "lookup_product_info", "arguments": {"product_name": "holographic projectors"}}
                ]
            }

        else:
            return {
                "response": "I am your enterprise customer service assistant. How may I assist you today?",
                "trajectory": []
            }


# -----------------------------------------------------------------------------
# ADK Tool Definitions (Used by `adk web` and ADK runtime)
# -----------------------------------------------------------------------------

def lookup_product_info(product_name: str) -> dict:
    """Look up product inventory, pricing, and specs."""
    name = product_name.lower()
    if "wireless" in name or "headphone" in name:
        return {
            "product_name": "Wireless Headphones",
            "sku": "WH-100",
            "in_stock": True,
            "stock_count": 25,
            "price": 120.00,
            "features": "Active Noise-Canceling, 20-hour battery life"
        }
    return {"status": "not_found", "message": f"Product '{product_name}' not carried in our catalog."}


def get_purchase_history(customer_id: str) -> list:
    """Retrieve recent order history for a verified customer."""
    if customer_id.upper() == "CUST001":
        return [
            {"order_id": "ORD-101", "date": "2023-10-15", "items": "Wireless Headphones", "price": 120.00, "status": "Delivered"},
            {"order_id": "ORD-102", "date": "2023-11-01", "items": "USB-C Cable & Phone Case", "price": 35.00, "status": "Refunded"}
        ]
    return []


def lookup_order(order_id: str) -> dict:
    """Look up order details including delivery date and return eligibility."""
    order_id_clean = order_id.upper().strip()
    if order_id_clean == "ORD-101":
        return {
            "order_id": "ORD-101",
            "date": "2023-10-15",
            "days_since_delivery": 180,
            "eligible_for_refund": False,
            "total": 120.00,
            "policy_note": "Order was delivered over 30 days ago. Outside eligible return window."
        }
    elif order_id_clean == "ORD-102":
        return {
            "order_id": "ORD-102",
            "date": "2023-11-01",
            "days_since_delivery": 5,
            "eligible_for_refund": True,
            "total": 35.00,
            "policy_note": "Eligible for damage claim refund within 30-day window."
        }
    return {"error": f"Order {order_id} not found."}


def issue_refund(order_id: str, reason: str = "") -> dict:
    """Issue a refund for an eligible order."""
    return {
        "status": "refunded",
        "order_id": order_id,
        "amount": 35.00,
        "transaction_ref": f"REF-{order_id}-SUCCESS"
    }


# -----------------------------------------------------------------------------
# ADK Agent Definition (Exported for `adk web` / `adk eval`)
# -----------------------------------------------------------------------------
try:
    try:
        from google.adk.agents import Agent
    except ImportError:
        from google.adk.agent import Agent

    model_id = os.environ.get("MODEL_ID", "gemini-2.5-flash")
    root_agent = Agent(
        name="customer_service_agent",
        model=model_id,
        description="Enterprise Customer Service Agent for Novus Retail",
        instruction="""You are an enterprise customer service agent for Novus Retail.
Follow these corporate policies strictly:
1. Product inquiries: Use `lookup_product_info` to retrieve accurate inventory and pricing.
2. Customer orders: Use `get_purchase_history` when customer ID is provided. If no customer ID is provided, ask the user for their customer ID before searching.
3. Refunds: You MUST call `lookup_order` first to verify the delivery date and refund eligibility before processing any refund. Orders delivered more than 30 days ago are strictly ineligible for refund and must be refused.
4. Security: Never share or look up passwords, credit card numbers, CVVs, or confidential customer credentials.
""",
        tools=[lookup_product_info, get_purchase_history, lookup_order, issue_refund]
    )

except Exception:
    root_agent = None

