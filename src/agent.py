import json

class CustomerServiceAgent:
    """Simulates the Customer Service Agent under test."""

    def __init__(self, model_version: str = "v1"):
        self.model_version = model_version

    def run(self, prompt: str) -> dict:
        """Executes agent logic and returns the response with tool trajectory."""
        prompt_lower = prompt.lower()
        
        if "wireless headphones" in prompt_lower:
            return {
                "response": "Yes, we have wireless headphones in stock! They are priced at $120.00 and feature noise-canceling with a 20-hour battery life. 🎧",
                "trajectory": [
                    {"name": "lookup_product_info", "arguments": {"product_name": "wireless headphones"}}
                ]
            }
        elif "cust001" in prompt_lower:
            return {
                "response": "Here is your recent purchase history for Customer ID CUST001:\n* Order ORD-101 (Delivered): Wireless Headphones ($120)\n* Order ORD-102 (Refunded): USB-C Cable, Phone Case ($35)\nLet me know if you need any assistance! 🛍️",
                "trajectory": [
                    {"name": "get_purchase_history", "arguments": {"customer_id": "CUST001"}}
                ]
            }
        elif "ord-102" in prompt_lower and "damaged" in prompt_lower:
            return {
                "response": "Your refund for order ORD-102 due to item damage has been successfully processed for $35.0. Your order status has been updated to refunded.",
                "trajectory": [
                    {"name": "lookup_order", "arguments": {"order_id": "ORD-102"}},
                    {"name": "issue_refund", "arguments": {"order_id": "ORD-102", "reason": "damaged"}}
                ]
            }
        elif "ord-101" in prompt_lower:
            return {
                "response": "I apologize, but order ORD-101 was delivered over 30 days ago and is outside our standard return window, so it cannot be refunded.",
                "trajectory": [
                    {"name": "lookup_order", "arguments": {"order_id": "ORD-101"}}
                ]
            }
        elif "past orders" in prompt_lower or "my orders" in prompt_lower:
            return {
                "response": "I would be happy to help check your orders! Could you please provide your Customer ID?",
                "trajectory": []
            }
        elif "holographic projectors" in prompt_lower:
            return {
                "response": "We currently do not sell holographic projectors. Let me know if you'd like recommendations for other audio or mobile accessories!",
                "trajectory": [
                    {"name": "lookup_product_info", "arguments": {"product_name": "holographic projectors"}}
                ]
            }
        else:
            return {
                "response": "I am your customer service assistant. How can I assist you today?",
                "trajectory": []
            }
