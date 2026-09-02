import json

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
