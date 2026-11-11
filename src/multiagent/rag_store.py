import os
import re
import difflib
from typing import Optional, List, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from .config import OPENAI_API_KEY, GEMINI_API_KEY, MODEL_PROVIDER, CHROMA_PERSIST_DIR

_vectorstore = None
_vectorstore_attempted = False

# ---------------------------------------------------------------------------
# Comprehensive Built-in E-Commerce & Customer Support Knowledge Base
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "question": "What is your return policy?",
        "answer": (
            "We offer a 30-day hassle-free return policy. Items must be unused, in their original packaging, "
            "and in the same condition you received them. Once your returned item arrives at our warehouse and is inspected, "
            "we will process your refund to your original payment method within 3 to 5 business days. Return shipping is free "
            "for damaged, defective, or incorrect items."
        )
    },
    {
        "question": "How do I return an item or start a return?",
        "answer": (
            "To initiate a return, navigate to our Returns portal or contact support with your Order ID (format: ORDxxxxx). "
            "We will generate a prepaid return shipping label for you. Simply pack the item securely, attach the label, "
            "and drop it off at any authorized courier location."
        )
    },
    {
        "question": "What payment methods do you support?",
        "answer": (
            "We support all major payment methods including Credit and Debit cards (Visa, MasterCard, American Express, Discover), "
            "PayPal, Apple Pay, Google Pay, and Shop Pay. We also offer buy-now-pay-later installment plans through Klarna and Afterpay. "
            "All transactions are securely encrypted with industry-standard 256-bit SSL encryption."
        )
    },
    {
        "question": "How long does shipping and delivery take?",
        "answer": (
            "We offer three shipping options: Standard Shipping (3-5 business days), Expedited Shipping (2-3 business days), "
            "and Priority Overnight (1 business day). Orders placed before 2:00 PM EST on business days are dispatched on the same day. "
            "Free standard shipping is automatically applied to all domestic orders over $50."
        )
    },
    {
        "question": "Do you offer international shipping?",
        "answer": (
            "Yes, we ship to over 60 countries worldwide. International shipping typically takes 7 to 14 business days depending on customs. "
            "All import duties, taxes, and fees are calculated upfront at checkout so there are no unexpected customs charges upon delivery."
        )
    },
    {
        "question": "Can I order without creating an account (guest checkout)?",
        "answer": (
            "Yes! You can complete your purchase as a guest without creating an account. Simply provide your email address and shipping "
            "details during checkout to receive order confirmation, tracking links, and digital receipts. Creating an account is optional "
            "but allows you to save addresses, track orders live, and view past purchases."
        )
    },
    {
        "question": "What should I do if my order arrives damaged, broken, or defective?",
        "answer": (
            "If your order arrives damaged, broken, or defective, please contact us within 7 days of delivery. "
            "Provide your Order ID (format: ORDxxxxx) and a brief description or photo of the damage. We will immediately arrange "
            "an expedited replacement shipment at no extra charge or process a 100% full refund to your original payment method."
        )
    },
    {
        "question": "How long does a refund take to process?",
        "answer": (
            "Once a refund is initiated by our support team or after your returned item is inspected, refunds are submitted immediately. "
            "The funds typically appear on your credit card or bank account in 3 to 5 business days, depending on your bank's processing schedule."
        )
    },
    {
        "question": "How can I cancel my order?",
        "answer": (
            "You can cancel your order within 1 hour of placing it, as long as it has not yet been processed by our warehouse. "
            "To cancel, reach out via this chat with your Order ID or click 'Cancel Order' on your order confirmation page. "
            "If the item has already shipped, you can simply refuse delivery or use our free return portal once it arrives."
        )
    },
    {
        "question": "How do I track my order status?",
        "answer": (
            "You can track your order at any time directly in this chat by typing your Order ID (e.g. ORD12345), or by clicking the "
            "tracking link sent in your shipping confirmation email. Courier status updates (FedEx, UPS, DHL) are updated in real-time."
        )
    },
    {
        "question": "Do your products come with a warranty?",
        "answer": (
            "All hardware and electronic products come with a complimentary 1-year manufacturer warranty that covers factory defects "
            "and component failures. We also offer optional 2-year and 3-year extended protection plans covering accidental drops and spills."
        )
    },
    {
        "question": "How do I change my shipping address?",
        "answer": (
            "If your order has not yet been shipped, our support team can update your delivery address immediately. Please provide your "
            "Order ID and the new shipping address. Once an order is with the courier, address changes must be requested directly through the courier's tracking portal."
        )
    },
    {
        "question": "Can I exchange an item for another size or color?",
        "answer": (
            "Yes, exchanges are free within 30 days of purchase. Contact our support team with your Order ID and the preferred replacement size or color. "
            "We will send you a prepaid return label and ship your replacement as soon as the carrier scans your return package."
        )
    },
    {
        "question": "How can I talk to a human customer service agent or representative?",
        "answer": (
            "Our automated AI assistant handles most order tracking, return, and policy questions 24/7. If you require human assistance, "
            "our senior support team is available Monday through Friday from 8:00 AM to 8:00 PM EST. Simply ask to 'speak to a human' or 'escalate ticket', "
            "and we will open an escalated ticket with a guaranteed response within 2-4 hours."
        )
    },
    {
        "question": "What should I do if my package says delivered but I did not receive it?",
        "answer": (
            "If your tracking shows delivered but you haven't received it, please check around your delivery location, with neighbors, or your building leasing office. "
            "Couriers occasionally mark packages delivered up to 24 hours before physical drop-off. If you still cannot locate it after 24 hours, contact us with your "
            "Order ID and we will initiate a courier investigation and provide a free replacement or full refund."
        )
    }
]

STOPWORDS = {
    "what", "is", "are", "your", "the", "a", "an", "i", "can", "do", "did", "does",
    "you", "my", "how", "to", "for", "in", "of", "and", "or", "on", "with", "if", "it",
    "there", "we", "our", "us", "this", "that", "tell", "me", "about"
}


def _score_faq_match(query: str, target_question: str) -> float:
    """Calculate hybrid similarity score between user query and FAQ question."""
    q_lower = query.lower().strip()
    t_lower = target_question.lower().strip()

    # Exact phrase sequence match
    seq_ratio = difflib.SequenceMatcher(None, q_lower, t_lower).ratio()

    # Meaningful keyword overlap
    q_words = [w for w in re.findall(r"\w+", q_lower) if w not in STOPWORDS]
    t_words = [w for w in re.findall(r"\w+", t_lower) if w not in STOPWORDS]
    if not q_words:
        q_words = re.findall(r"\w+", q_lower)
    if not t_words:
        t_words = re.findall(r"\w+", t_lower)

    overlap = len(set(q_words) & set(t_words))
    keyword_recall = overlap / len(set(q_words)) if q_words else 0.0
    keyword_jaccard = overlap / len(set(q_words) | set(t_words)) if (set(q_words) | set(t_words)) else 0.0

    return (seq_ratio * 0.35) + (keyword_recall * 0.45) + (keyword_jaccard * 0.20)


def _search_knowledge_base_fallback(query: str, top_k: int = 3) -> List[str]:
    """Search built-in FAQ knowledge base using precision hybrid scoring."""
    scored = []
    for row in KNOWLEDGE_BASE:
        score = _score_faq_match(query, row["question"])
        if score >= 0.15:  # Realistic matching threshold
            scored.append((score, f"Question: {row['question']}\nAnswer: {row['answer']}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def get_embeddings():
    """Return appropriate embeddings model based on configuration."""
    if OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=OPENAI_API_KEY
            )
        except Exception as e:
            print(f"[RAG] OpenAIEmbeddings init failed: {e}")

    if GEMINI_API_KEY and MODEL_PROVIDER == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY
            )
        except Exception as e:
            print(f"[RAG] Gemini Embeddings init failed: {e}")

    return None


def get_vectorstore():
    """
    Initialize, seed, and return Chroma Vector DB.
    """
    global _vectorstore, _vectorstore_attempted
    if _vectorstore is not None:
        return _vectorstore
    if _vectorstore_attempted:
        return None

    embeddings = get_embeddings()
    if not embeddings:
        print("[RAG] Chroma Vector DB: Embeddings not configured. Using high-performance in-memory FAQ engine.")
        _vectorstore_attempted = True
        return None

    try:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # 1. Try loading existing Chroma Vector DB
        try:
            candidate = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
            count = candidate._collection.count()
            if count >= len(KNOWLEDGE_BASE):
                _vectorstore = candidate
                print(f"[RAG] Chroma Vector DB: YES (Loaded {count} documents from disk)")
                _vectorstore_attempted = True
                return _vectorstore
        except Exception as load_err:
            print(f"[RAG] Existing Chroma check: {load_err}")

        # 2. Seed Chroma Vector DB with curated Knowledge Base
        print(f"[RAG] Seeding Chroma Vector DB with {len(KNOWLEDGE_BASE)} knowledge documents...")
        documents = [
            Document(
                page_content=f"Question: {r['question']}\nAnswer: {r['answer']}",
                metadata={"topic": r["question"]}
            )
            for r in KNOWLEDGE_BASE
        ]

        _vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = _vectorstore._collection.count()
        print(f"[RAG] Chroma Vector DB: Initialized and populated with {count} vectors.")
        _vectorstore_attempted = True
        return _vectorstore

    except Exception as e:
        print(f"[RAG] Chroma Vector DB initialization failed: {e}. Falling back to in-memory matching.")
        _vectorstore = None
        _vectorstore_attempted = True
        return None


@tool
def search_faq(query: str) -> str:
    """
    Search company FAQs, return policies, shipping guides, payment methods, and warranty information.
    Returns retrieved FAQ context or 'NO_RELEVANT_FAQ_FOUND'.
    """
    clean_query = query.strip()
    print(f"[RAG] Query: {clean_query}")

    retrieved_docs: List[str] = []

    # 1. First try Chroma Vector DB
    vs = get_vectorstore()
    if vs is not None:
        try:
            results = vs.similarity_search(clean_query, k=3)
            if results:
                retrieved_docs = [doc.page_content for doc in results if doc.page_content.strip()]
                print(f"[RAG] Chroma Vector DB matched {len(retrieved_docs)} documents.")
        except Exception as e:
            print(f"[RAG] Chroma similarity search failed: {e}")

    # 2. Fallback to built-in knowledge base if Chroma returned no results or was unavailable
    if not retrieved_docs:
        retrieved_docs = _search_knowledge_base_fallback(clean_query, top_k=3)
        if retrieved_docs:
            print(f"[RAG] Knowledge Base matched {len(retrieved_docs)} documents.")

    if retrieved_docs:
        return "\n---\n".join(retrieved_docs)

    return "NO_RELEVANT_FAQ_FOUND"
