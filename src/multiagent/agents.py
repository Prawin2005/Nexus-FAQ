import re
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from .state import AgentState
from .config import (
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    OPENAI_MODEL_NAME,
    GEMINI_MODEL_NAME
)
from .tools import get_order_details, process_refund, escalate_complaint
from .rag_store import search_faq


def get_latest_user_query(messages: List[BaseMessage]) -> str:
    """Extract the content of the most recent HumanMessage in the conversation."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            content = msg.content
            return content.strip() if isinstance(content, str) else str(content).strip()
    return ""


def find_order_id_in_history(messages: List[BaseMessage]) -> Optional[str]:
    """Search backwards through human messages ONLY for a valid order ID pattern."""
    for msg in reversed(messages):
        # Only inspect messages sent by the HUMAN user, never AI assistant messages!
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            matches = re.findall(r"\bORD\d+\b", content.upper())
            if matches:
                return matches[0]
    return None


def get_llm():
    """
    Returns ChatOpenAI or ChatGoogleGenerativeAI instance based on configuration.
    """
    if MODEL_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            print("[CONFIG] WARNING: Provider is 'openai' but OPENAI_API_KEY is not set.")
            return None
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=OPENAI_MODEL_NAME,
                openai_api_key=OPENAI_API_KEY,
                temperature=0.3,
                max_retries=1
            )
        except Exception as e:
            print(f"[LLM] Error initializing ChatOpenAI ({OPENAI_MODEL_NAME}): {e}")
            return None

    if MODEL_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            print("[CONFIG] WARNING: Provider is 'gemini' but GEMINI_API_KEY is not set.")
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GEMINI_API_KEY,
                temperature=0.3
            )
        except Exception as e:
            print(f"[LLM] Error initializing ChatGoogleGenerativeAI ({GEMINI_MODEL_NAME}): {e}")
            return None

    return None


def get_active_model_name() -> str:
    """Return the active model name for logging."""
    if MODEL_PROVIDER == "openai":
        return f"OpenAI ({OPENAI_MODEL_NAME})"
    elif MODEL_PROVIDER == "gemini":
        return f"Gemini ({GEMINI_MODEL_NAME})"
    return "None (Rule-based Fallback)"


# 1. Query Router Node
def query_router_node(state: AgentState) -> Dict[str, Any]:
    """
    Query Router Node: Categorizes customer query into faq_docs, order_query, complaint, greeting, or off_topic.
    """
    messages = state.get("messages", [])
    user_query = get_latest_user_query(messages)
    q_lower = user_query.lower()

    print(f"[ROUTER] User query: {user_query}")

    # 1. Check greeting
    words = [w.strip("!?,.") for w in q_lower.split()]
    greeting_words = {"hi", "hello", "hey", "hola", "howdy", "greetings"}
    is_greeting = any(w in greeting_words for w in words) or \
                  any(p in q_lower for p in ["good morning", "good afternoon", "good evening", "good day"])

    # 2. Check order ID pattern
    has_order_id = bool(re.search(r"\bORD\d+\b", user_query.upper()))

    # Check for order ID follow-up / clarification phrases (e.g. "hey i didnt sent you the id yet")
    order_id_followup_phrases = [
        "didn't send", "did not send", "haven't sent", "havent sent", "not sent",
        "didn't give", "did not give", "haven't provided", "havent provided",
        "the id", "my id", "order id", "which id", "which order", "wrong order",
        "no id", "gave you the id", "sent you the id"
    ]
    is_id_followup = any(p in q_lower for p in order_id_followup_phrases)

    # 3. Check specific e-commerce / customer support keywords
    ecommerce_keywords = [
        "order", "track", "delivery", "shipping", "package", "parcel", "shipment",
        "arrive", "arrival", "delay", "courier", "return", "refund", "damaged", "broken",
        "defective", "faulty", "warranty", "guarantee", "payment", "card", "invoice",
        "receipt", "billing", "bill", "charge", "account", "login", "password", "sign up",
        "cancel", "product", "item", "purchase", "bought", "cart", "checkout", "store",
        "price", "discount", "coupon", "promo", "voucher", "support", "help", "customer service",
        "policy", "policies", "contact", "representative", "id", "order id"
    ]
    is_ecommerce = any(k in q_lower for k in ecommerce_keywords) or has_order_id or is_id_followup

    # 4. Check explicit complaints & refund issues
    complaint_phrases = [
        "refund", "money back", "damaged", "broken", "cracked", "defective", "faulty",
        "wrong item", "wrong product", "missing item", "missing product", "never arrived",
        "lost package", "stolen package", "file a complaint", "make a complaint",
        "bad service", "terrible service", "horrible service", "poor service",
        "bad experience", "terrible experience", "horrible experience", "unacceptable",
        "double charged", "charged twice", "overcharged", "talk to supervisor", "talk to manager",
        "escalate", "dispute"
    ]
    is_complaint = (any(p in q_lower for p in complaint_phrases) and is_ecommerce) or (is_id_followup and state.get("intent") == "complaint")

    # 5. Order tracking vs ordering FAQ
    tracking_phrases = ["track", "where is", "where's", "order status", "my package", "delivery time", "how long will delivery", "when will", "delay", "shipping status"]
    is_tracking = has_order_id or any(p in q_lower for p in tracking_phrases) or ("my order" in q_lower and not any(f in q_lower for f in ["how", "can i", "policy", "cancel"])) or (is_id_followup and state.get("intent") == "order_query")

    faq_ordering_phrases = ["without creating an account", "without an account", "can i order", "how to order", "how can i order", "order as guest", "guest checkout", "cancel an order"]
    is_ordering_faq = any(p in q_lower for p in faq_ordering_phrases) and not has_order_id

    # Classification logic
    if is_greeting and len(words) <= 4:
        intent = "greeting"
        reasoning = "Query detected as a standard greeting."
    elif is_id_followup:
        # Route ID clarification to complaint or order_query based on prior context or default to support
        intent = "complaint" if state.get("intent") == "complaint" else "order_query"
        reasoning = "Query detected as an Order ID follow-up / clarification."
    elif not is_ecommerce and not is_greeting:
        intent = "off_topic"
        reasoning = "Query is not related to products, orders, policies, or customer support."
    elif is_complaint:
        intent = "complaint"
        reasoning = "Query detected as Customer Complaint / Refund / Support Request."
    elif is_ordering_faq:
        intent = "faq_docs"
        reasoning = "Query detected as an Ordering Policy / FAQ Inquiry."
    elif is_tracking:
        intent = "order_query"
        reasoning = "Query detected as Order Tracking & Details Request."
    else:
        intent = "faq_docs"
        reasoning = "Query detected as FAQ, Policy, or Knowledgebase Request."

    print(f"[ROUTER] Intent: {intent}")

    log_entry = {
        "node": "Query Router",
        "action": f"Classified intent as '{intent}'",
        "reasoning": reasoning
    }

    return {
        "intent": intent,
        "agent_logs": [log_entry]
    }


# 2. RAG Agent Node (FAQ / Docs Branch)
def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    RAG Agent Node: Uses LLM + FAQ context to answer policy and general questions intelligently.
    """
    print("[AGENT] Selected agent: RAG Agent")
    messages = state.get("messages", [])
    user_query = get_latest_user_query(messages)

    # Retrieve relevant FAQ docs
    faq_context = search_faq.invoke({"query": user_query})

    llm = get_llm()
    model_name = get_active_model_name()
    print(f"[LLM] Model: {model_name}")

    draft = None
    if llm:
        system_prompt = (
            "You are a helpful customer support assistant.\n"
            "Answer the customer's CURRENT question directly, concisely, and accurately using the provided FAQ context.\n"
            "Rules:\n"
            "- Give ONLY the direct answer to the customer's specific question.\n"
            "- Do NOT output multiple unrelated FAQ questions or repeat the entire knowledge base.\n"
            "- Do NOT repeat 'Question:' or 'Answer:' labels.\n"
            "- If the FAQ context does not contain the answer, say that the information is unavailable and ask an appropriate follow-up question.\n\n"
            f"FAQ Context:\n{faq_context}"
        )
        try:
            # Build conversation history for LLM (last 4 messages)
            llm_messages = [SystemMessage(content=system_prompt)]
            recent_msgs = messages[-4:] if len(messages) > 4 else messages
            for msg in recent_msgs:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    llm_messages.append(msg)
            if not any(isinstance(m, HumanMessage) and m.content == user_query for m in llm_messages):
                llm_messages.append(HumanMessage(content=user_query))

            response = llm.invoke(llm_messages)
            draft = response.content
            print(f"[LLM] Response generated: {draft[:60]}...")
        except Exception as e:
            print(f"[LLM] Error during invoke: {e}")
            draft = None

    # Deterministic, exact response if LLM is unavailable or failed
    if not draft:
        if faq_context and faq_context != "NO_RELEVANT_FAQ_FOUND":
            # Extract the exact answer of the best matching FAQ item
            m = re.search(r"Answer:\s*(.+?)(?:\n---|\Z)", faq_context, re.DOTALL)
            if m:
                draft = m.group(1).strip()
            else:
                draft = faq_context.strip()
        else:
            draft = (
                f"I could not find specific FAQ details in our knowledge base regarding '{user_query}'. "
                f"Could you please rephrase your question, or would you like me to connect you with our support team?"
            )
        print(f"[LLM] Response generated (fallback): {draft[:60]}...")

    log_entry = {
        "node": "RAG Agent",
        "action": "Queried FAQ Vector Store + LLM synthesis",
        "draft": draft
    }

    return {
        "draft_response": draft,
        "agent_logs": [log_entry]
    }


# 2b. Greeting Agent Node
def greeting_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Greeting Agent Node: Responds warmly and contextually to greetings.
    """
    print("[AGENT] Selected agent: Greeting Agent")
    messages = state.get("messages", [])
    user_query = get_latest_user_query(messages) or "Hello"

    llm = get_llm()
    model_name = get_active_model_name()
    print(f"[LLM] Model: {model_name}")

    draft = None
    if llm:
        system_prompt = (
            "You are a warm, professional customer support agent for an e-commerce company. "
            "The customer has just greeted you. Respond with a friendly welcome, "
            "briefly mention what you can help with (order tracking, returns, refunds, product questions), "
            "and ask how you can assist them today. Keep it concise — 2-3 sentences max."
        )
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ])
            draft = response.content
            print(f"[LLM] Response generated: {draft[:60]}...")
        except Exception as e:
            print(f"[LLM] Error during greeting invoke: {e}")
            draft = None

    if not draft:
        draft = (
            "Hello! Welcome to our customer support. I'm here to help you with order tracking, "
            "returns, refunds, and policy questions. How can I assist you today?"
        )
        print(f"[LLM] Response generated (fallback): {draft[:60]}...")

    log_entry = {
        "node": "Greeting Agent",
        "action": "Responded to greeting",
        "draft": draft
    }

    return {
        "draft_response": draft,
        "agent_logs": [log_entry]
    }


# 3. Order Tool Node (Order Query Branch)
def order_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Order Tool Node: Fetches order details and answers order queries contextually.
    """
    print("[AGENT] Selected agent: Order Agent")
    messages = state.get("messages", [])
    user_query = get_latest_user_query(messages)

    # Check for order ID in current query, or fallback to conversation history
    order_id = None
    order_ids = re.findall(r"ORD\d+", user_query.upper())
    if order_ids:
        order_id = order_ids[0]
    else:
        order_id = find_order_id_in_history(messages)

    if order_id:
        order_info = get_order_details.invoke({"order_id": order_id})
        tool_context = f"Order lookup for {order_id}:\n{order_info}"
    else:
        order_info = None
        tool_context = "No order ID was found in the customer message or conversation history."

    llm = get_llm()
    model_name = get_active_model_name()
    print(f"[LLM] Model: {model_name}")

    draft = None
    if llm:
        system_prompt = (
            "You are a helpful customer support agent for an e-commerce company.\n"
            "STRICT RULES:\n"
            "- If the customer provided an Order ID and order info is in the context below, answer their question clearly.\n"
            "- If no Order ID was provided, you MUST NOT invent, guess, or assume an order ID. "
            "Politely ask the customer to provide their Order ID (format: ORDxxxxx).\n\n"
            f"Order Context:\n{tool_context}"
        )
        try:
            llm_messages = [SystemMessage(content=system_prompt)]
            recent_msgs = messages[-4:] if len(messages) > 4 else messages
            for msg in recent_msgs:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    llm_messages.append(msg)
            if not any(isinstance(m, HumanMessage) and m.content == user_query for m in llm_messages):
                llm_messages.append(HumanMessage(content=user_query))

            response = llm.invoke(llm_messages)
            draft = response.content
            print(f"[LLM] Response generated: {draft[:60]}...")
        except Exception as e:
            print(f"[LLM] Error during order invoke: {e}")
            draft = None

    if not draft:
        if order_id and order_info and "not found" not in order_info.lower():
            q_lower = user_query.lower()
            if "delivery" in q_lower or "how long" in q_lower or "when" in q_lower:
                draft = (
                    f"Regarding order {order_id}: {order_info}. "
                    f"Standard shipping typically takes 2-3 business days."
                )
            elif "not arrive" in q_lower or "doesn't arrive" in q_lower or "lost" in q_lower:
                draft = (
                    f"If order {order_id} does not arrive as scheduled, please notify us immediately. "
                    f"We offer full coverage with an expedited replacement or a 100% refund."
                )
            else:
                draft = f"Here is the status for your order:\n\n{order_info}\n\nPlease let me know if you need help with anything else!"
        elif order_id and order_info:
            draft = f"{order_info} Please double-check the Order ID (e.g. ORD12345) and try again."
        else:
            draft = "Please provide your Order ID (for example, ORD12345) so I can check the status and details of your order."
        print(f"[LLM] Response generated (fallback): {draft[:60]}...")

    log_entry = {
        "node": "Order Agent",
        "action": "Processed order query",
        "draft": draft
    }

    return {
        "draft_response": draft,
        "agent_logs": [log_entry]
    }


# 4. Support Agent Node (Complaint / Refund Branch)
def support_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Support Agent Node: Handles complaints, damaged items, refunds, and escalations contextually.
    """
    print("[AGENT] Selected agent: Support/Complaint Agent")
    messages = state.get("messages", [])
    user_query = get_latest_user_query(messages)
    q_lower = user_query.lower()

    # Check if the customer provided an Order ID in the CURRENT message
    order_ids = re.findall(r"\bORD\d+\b", user_query.upper())
    # Only fall back to history if the user explicitly references an order and human messages provided one
    has_explicit_history_ref = any(w in q_lower for w in ["that order", "this order", "the previous order", "my previous order"])
    if order_ids:
        order_id = order_ids[0]
    elif has_explicit_history_ref:
        order_id = find_order_id_in_history(messages)
    else:
        order_id = None

    # Check if user is clarifying that they have NOT provided an ID yet
    is_clarifying_no_id = any(p in q_lower for p in [
        "didn't send", "did not send", "haven't sent", "havent sent",
        "didn't give", "did not give", "haven't provided", "havent provided",
        "no id", "not sent"
    ])

    tool_results = []
    if is_clarifying_no_id:
        tool_context = "Customer clarified they have not provided an Order ID yet. Apologize for any misunderstanding and ask for their Order ID."
    elif "refund" in q_lower and order_id:
        res = process_refund.invoke({"order_id": order_id, "amount": 49.99, "reason": user_query})
        tool_results.append(f"Refund Action: {res}")
        tool_context = "\n".join(tool_results)
    elif order_id:
        order_info = get_order_details.invoke({"order_id": order_id})
        tool_results.append(f"Order Context: {order_info}")
        tool_context = "\n".join(tool_results)
    else:
        tool_context = "CRITICAL: No Order ID has been provided by the customer yet. Do NOT invent, assume, or process any order. Politely ask the customer for their Order ID (format: ORDxxxxx)."

    llm = get_llm()
    model_name = get_active_model_name()
    print(f"[LLM] Model: {model_name}")

    draft = None
    if llm:
        system_prompt = (
            "You are an empathetic, professional customer support specialist handling complaints and refund requests.\n"
            "STRICT OPERATING RULES:\n"
            "1. If the customer indicates they have not sent an Order ID yet, apologize politely and request their Order ID.\n"
            "2. If NO Order ID was provided by the customer, you MUST NOT invent, guess, or assume an Order ID like ORD12345. "
            "You MUST ask the customer for their Order ID (format: ORDxxxxx) and briefly explain why it is needed.\n"
            "3. NEVER claim a refund has been initiated, processed, or submitted unless confirmed in the Support Tool Context below.\n"
            "4. Acknowledge customer frustration with genuine empathy, take accountability, and outline clear next steps.\n\n"
            f"Support Tool Context:\n{tool_context}"
        )
        try:
            llm_messages = [SystemMessage(content=system_prompt)]
            recent_msgs = messages[-4:] if len(messages) > 4 else messages
            for msg in recent_msgs:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    llm_messages.append(msg)
            if not any(isinstance(m, HumanMessage) and m.content == user_query for m in llm_messages):
                llm_messages.append(HumanMessage(content=user_query))

            response = llm.invoke(llm_messages)
            draft = response.content
            print(f"[LLM] Response generated: {draft[:60]}...")
        except Exception as e:
            print(f"[LLM] Error during support invoke: {e}")
            draft = None

    if not draft:
        if is_clarifying_no_id:
            draft = (
                "I apologize for the misunderstanding! Please share your Order ID (format: ORDxxxxx) "
                "whenever you're ready, and I'll immediately look up the details to help you with your refund or replacement."
            )
        elif not order_id:
            if "damaged" in q_lower or "broken" in q_lower:
                draft = (
                    "I am very sorry to hear that your order arrived damaged. "
                    "To process an immediate replacement or full refund for you, could you please provide your Order ID (format: ORDxxxxx)?"
                )
            elif "refund" in q_lower:
                draft = (
                    "I'd be glad to help you with your refund request. "
                    "Could you please provide your Order ID (format: ORDxxxxx) so I can verify the purchase and initiate your refund?"
                )
            else:
                draft = (
                    "I apologize for the inconvenience with your order. "
                    "Could you please share your Order ID so our support team can pull up your account and resolve this for you?"
                )
        elif "refund" in q_lower and order_id and any("Refund Action" in r for r in tool_results):
            draft = (
                f"I'm very sorry to hear about the issue with your order {order_id}. "
                f"I have initiated a refund request on your behalf. "
                f"Refund Status: Successfully submitted for processing. You should see the credit returned to your original payment method in 3-5 business days."
            )
        elif tool_results:
            draft = f"Thank you for contacting support regarding your issue.\n\n{tool_context}\n\nOur team is actively assisting you."
        else:
            esc_res = escalate_complaint.invoke({"issue_summary": user_query, "customer_id": state.get("customer_id", "CUST101")})
            ticket_info = esc_res if isinstance(esc_res, str) else str(esc_res)
            draft = (
                f"I apologize for the inconvenience caused. "
                f"Your complaint has been escalated to our senior support team. {ticket_info} "
                f"Please share your Order ID (format: ORDxxxxx) so we can expedite your resolution."
            )
        print(f"[LLM] Response generated (fallback): {draft[:60]}...")

    log_entry = {
        "node": "Support Agent",
        "action": "Handled complaint/refund request",
        "draft": draft
    }
    return {
        "draft_response": draft,
        "agent_logs": [log_entry]
    }


# 5. Off-Topic Agent Node
def off_topic_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Off-Topic Agent Node: Handles queries unrelated to e-commerce or support.
    Returns a polite refusal indicating the chatbot only answers product/service questions.
    """
    print("[AGENT] Selected agent: Off-Topic Agent")
    draft = "Sorry, I can only answer questions related to products and services."
    log_entry = {
        "node": "Off-Topic Agent",
        "action": "Handled off-topic input",
        "draft": draft,
    }
    return {
        "draft_response": draft,
        "agent_logs": [log_entry],
    }


# 6. Validator Node
def validator_node(state: AgentState) -> Dict[str, Any]:
    """
    Validator Node: Inspects draft response to ensure quality, accuracy, anti-hallucination, and tone.
    """
    draft = state.get("draft_response", "") or ""
    messages = state.get("messages", [])
    user_query = get_latest_user_query(messages)

    is_valid = True
    feedback = "Approved: Response is helpful, polite, and directly addresses the customer query."
    validated_text = draft

    # 1. Check for empty or too short draft
    if not draft or len(draft.strip()) < 10:
        is_valid = False
        feedback = "Revised: Draft was incomplete or too short."
        validated_text = "Hello! Thank you for reaching out. How can I assist you with your orders, returns, or support questions today?"

    # 2. Anti-Hallucination Guardrail: Did the draft invent an Order ID the customer never provided?
    draft_order_ids = re.findall(r"\bORD\d+\b", draft.upper())
    if draft_order_ids and is_valid:
        # Collect all order IDs actually provided by the HUMAN user in query or history
        user_order_ids = set(re.findall(r"\bORD\d+\b", user_query.upper()))
        history_order_id = find_order_id_in_history(messages)
        if history_order_id:
            user_order_ids.add(history_order_id.upper())

        # If draft asserts an order ID that the human user never gave
        hallucinated_ids = [oid for oid in draft_order_ids if oid not in user_order_ids]
        if hallucinated_ids:
            is_valid = False
            feedback = f"Hallucination Intercepted: Draft referenced unverified Order ID(s) {hallucinated_ids} not provided by customer."
            validated_text = (
                "I am very sorry to hear about the issue with your order. "
                "Could you please share your Order ID (format: ORDxxxxx) so I can look up the purchase details and assist you with your request?"
            )

    validation_status = "APPROVED" if is_valid else "REVISED"
    print(f"[VALIDATOR] Response validated: {validation_status} - {feedback}")

    log_entry = {
        "node": "Validator Node",
        "action": f"Validation status: {validation_status}",
        "feedback": feedback
    }

    return {
        "validated_response": validated_text,
        "validation_status": validation_status,
        "validator_feedback": feedback,
        "messages": [AIMessage(content=validated_text)],
        "agent_logs": [log_entry]
    }
