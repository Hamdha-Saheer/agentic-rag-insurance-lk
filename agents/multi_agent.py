# agents/multi_agent.py
import os
import sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Ensure parent directory is in path for configuration parameters
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.rag_pipeline import rag_query
from agents.domain_router import route_domain


# ── 1. LANGGRAPH STATE DEFINITION ─────────────────────────
class AgentState(TypedDict):
    question:       str
    user_profile:   dict
    active_domains: List[str]
    motor_answer:   str
    health_answer:  str
    life_answer:    str
    general_answer: str
    motor_sources:  List[str]
    health_sources: List[str]
    life_sources:   List[str]
    general_sources:List[str]
    final_answer:   str
    all_sources:    List[str]
    domains_used:   List[str]


# ── 2. SUPERVISOR NODE (ORCHESTRATION & ROUTING) ──────────
def supervisor_agent(state: AgentState) -> dict:
    """
    Analyzes the incoming question and determines which agent nodes to trigger.
    Enables multi-domain activation for cross-cutting insurance scenarios.
    """
    q       = state['question'].lower()
    profile = state.get('user_profile', {})
    
    # Extract primary structural domain
    primary = route_domain(state['question'], profile)
    domains = [primary]

    # Cross-domain keyword signals to trigger a secondary helper agent
    signals = {
        'motor':   ['vehicle', 'car', 'bike', 'accident', 'motor', 'traffic', 'road', 'driving', 'collision'],
        'health':  ['hospital', 'medical', 'agrahara', 'health', 'treatment', 'surgery', 'doctor', 'illness', 'icu'],
        'life':    ['life insurance', 'death benefit', 'beneficiary', 'long-term', 'endowment', 'life cover', 'nominee'],
        'general': ['travel', 'student', 'suraksha', 'property', 'fire', 'theft', 'flood', 'overseas', 'baggage'],
    }
    
    # Check for supplementary cross-domain matches
    for domain, kws in signals.items():
        if domain not in domains and any(k in q for k in kws):
            domains.append(domain)

    active_scoped = domains[:2]
    print(f'   [Supervisor] Primary Route: {primary.upper()} | Active Nodes: {active_scoped}')
    
    # Returns a state update dict to follow immutable state principles
    return {"active_domains": active_scoped}


# ── 3. INDIVIDUAL DOMAIN AGENT NODES ───────────────────────
def motor_agent(state: AgentState) -> dict:
    """Handles vehicle asset and third-party motor liability retrieval passes."""
    if 'motor' in state['active_domains']:
        print('   [Motor Agent] Actively querying motor vector index...')
        r = rag_query(state['question'], 'motor', state.get('user_profile', {}))
        return {"motor_answer": r['answer'], "motor_sources": r['sources']}
    return {"motor_answer": "", "motor_sources": []}


def health_agent(state: AgentState) -> dict:
    """Handles national medical schemes (Agrahara, SEMI, Pensioner health covers)."""
    if 'health' in state['active_domains']:
        print('   [Health Agent] Actively querying health vector index...')
        r = rag_query(state['question'], 'health', state.get('user_profile', {}))
        return {"health_answer": r['answer'], "health_sources": r['sources']}
    return {"health_answer": "", "health_sources": []}


def life_agent(state: AgentState) -> dict:
    """Handles long-term policies, death benefits, and maturity endowment structures."""
    if 'life' in state['active_domains']:
        print('   [Life Agent] Actively querying life vector index...')
        r = rag_query(state['question'], 'life', state.get('user_profile', {}))
        return {"life_answer": r['answer'], "life_sources": r['sources']}
    return {"life_answer": "", "life_sources": []}


def general_agent(state: AgentState) -> dict:
    """Handles Suraksha student policies, commercial property, and travel covers."""
    if 'general' in state['active_domains']:
        print('   [General Agent] Actively querying general vector index...')
        r = rag_query(state['question'], 'general', state.get('user_profile', {}))
        return {"general_answer": r['answer'], "general_sources": r['sources']}
    return {"general_answer": "", "general_sources": []}


# ── 4. CONTEXT AGGREGATOR NODE ────────────────────────────
def aggregator_agent(state: AgentState) -> dict:
    """
    Consolidates collected context segments from all activated domain streams.
    Formats responses cleanly using markdown separation rules.
    """
    parts   = []
    sources = []
    used    = []

    if state.get('motor_answer'):
        parts.append('### Motor Insurance\n' + state['motor_answer'].strip())
        sources += state.get('motor_sources', [])
        used.append('motor')

    if state.get('health_answer'):
        parts.append('### Health Insurance\n' + state['health_answer'].strip())
        sources += state.get('health_sources', [])
        used.append('health')

    if state.get('life_answer'):
        parts.append('### Life Insurance\n' + state['life_answer'].strip())
        sources += state.get('life_sources', [])
        used.append('life')

    if state.get('general_answer'):
        parts.append('### General Insurance\n' + state['general_answer'].strip())
        sources += state.get('general_sources', [])
        used.append('general')

    # Synthesize final answer payload
    if len(parts) == 0:
        final_ans = 'No relevant insurance policy information could be located in our repositories.'
    elif len(parts) == 1:
        final_ans = parts[0].split('\n', 1)[1].strip()
    else:
        final_ans = (
            'This scenario crosses multiple Sri Lankan insurance domains. '
            'Below is the aggregated advice compiled from relevant framework parameters:\n\n'
            + '\n\n---\n\n'.join(parts)
        )

    return {
        "final_answer": final_ans,
        "all_sources": list(set(sources)),
        "domains_used": used
    }


# ── 5. CONDITIONAL ROUTING LOGIC ──────────────────────────
def router_conditional_edge(state: AgentState) -> List[str]:
    """
    Tells LangGraph exactly which nodes to branch out to in parallel.
    """
    return state["active_domains"] if state["active_domains"] else ["general"]


# ── 6. GRAPH ENGINE BUILDER (PARALLEL FORK-JOIN TOPOLOGY) ─
def build_graph():
    """
    Compiles an optimized parallel-processing execution topology.
    """
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node('supervisor', supervisor_agent)
    g.add_node('motor',      motor_agent)
    g.add_node('health',     health_agent)
    g.add_node('life',       life_agent)
    g.add_node('general',    general_agent)
    g.add_node('aggregator', aggregator_agent)

    # Set up entry points and structural boundaries
    g.set_entry_point('supervisor')
    
    # Conditional Fork: Diverges concurrently to selected domains
    g.add_conditional_edges(
        'supervisor',
        router_conditional_edge,
        {
            'motor': 'motor',
            'health': 'health',
            'life': 'life',
            'general': 'general'
        }
    )
    
    # Join Boundary: Merges asynchronous paths straight back to aggregator
    g.add_edge('motor',      'aggregator')
    g.add_edge('health',     'aggregator')
    g.add_edge('life',       'aggregator')
    g.add_edge('general',    'aggregator')
    
    g.add_edge('aggregator', END)

    return g.compile()


# ── 7. MAIN SYSTEM WRAPPER INTERFACE ───────────────────────
_graph = None

def multi_agent_query(question: str, user_profile: dict = None) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state = {
        'question':        question,
        'user_profile':    user_profile or {},
        'active_domains':  [],
        'motor_answer':    '',
        'health_answer':   '',
        'life_answer':     '',
        'general_answer':  '',
        'motor_sources':   [],
        'health_sources':  [],
        'life_sources':    [],
        'general_sources': [],
        'final_answer':    '',
        'all_sources':     [],
        'domains_used':    []
    }

    result = _graph.invoke(initial_state)

    return {
        'answer':  result['final_answer'],
        'sources': result['all_sources'],
        'domains': result['domains_used'],
        'domain':  result['domains_used'][0] if result['domains_used'] else 'general',
    }


# --- Functional Verification Execution ---
if __name__ == "__main__":
    print("Testing Multi-Agent Orchestration Graph (Parallel Topology)...")
    test_question = "I had a motor accident and was admitted to the hospital. How do I file a claim?"
    test_profile = {"job": "Government Employee", "vehicle": "Car"}
    
    response = multi_agent_query(test_question, test_profile)
    print("\n--- TEST RESPONSE OUTPUT ---")
    print(response['answer'])
