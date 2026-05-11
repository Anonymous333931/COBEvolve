# ── FILE: cobol_moderniser/tests/test_step3.py ────────────────
"""
test_step3.py — Tests for MAPE-K planning controller behaviour.
Run: pytest tests/test_step3.py -v
"""
import sys, os, json, pytest
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from utils.cobol_parser import parse_cobol_source
from utils.graph_utils  import build_dependency_graph
from agents.comprehension_agent import assemble_knowledge_graph
from agents.planning_agent import (
    score_module, order_modules, plan, serialise_plan,
    load_plan, summarise_plan, ScoredModule, MigrationPlan
)

# ── Shared test data ────────────────────────────────────
ACCTPROC_SRC = '''
IDENTIFICATION DIVISION.
PROGRAM-ID. ACCTPROC.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 WS-BALANCE        PIC 9(8)V99 VALUE 0.
01 WS-MIN-BALANCE    PIC 9(7)V99 VALUE 100.00.
01 WS-MAX-TRANSACTION PIC 9(7)V99 VALUE 50000.00.
01 WS-RESULT-CODE    PIC 9(2)    VALUE 0.
PROCEDURE DIVISION.
MAIN-LOGIC.
    PERFORM VALIDATE-TRANSACTION.
    STOP RUN.
VALIDATE-TRANSACTION.
    IF WS-BALANCE < WS-MIN-BALANCE
        MOVE 1 TO WS-RESULT-CODE
    END-IF.
'''

INVMGMT_SRC = '''
IDENTIFICATION DIVISION.
PROGRAM-ID. INVMGMT.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 WS-PRODUCT-ID     PIC X(8).
01 WS-QUANTITY       PIC 9(5) VALUE 0.
01 WS-REORDER-LEVEL  PIC 9(4) VALUE 10.
PROCEDURE DIVISION.
MAIN-LOGIC.
    PERFORM CHECK-REORDER.
    CALL 'ACCTPROC' USING WS-PRODUCT-ID.
    STOP RUN.
CHECK-REORDER.
    IF WS-QUANTITY <= WS-REORDER-LEVEL
        DISPLAY 'REORDER REQUIRED'
    END-IF.
'''

@pytest.fixture
def kg():
    progs = [
        parse_cobol_source(ACCTPROC_SRC, 'ACCTPROC.cbl'),
        parse_cobol_source(INVMGMT_SRC,  'INVMGMT.cbl'),
    ]
    g = build_dependency_graph(progs)
    return assemble_knowledge_graph(progs, g, extract_rules=False)


# ── Scoring tests ──────────────────────────────────────
class TestRVFScoring:
    def test_scores_are_in_range(self, kg):
        graph = kg['_graph']
        from utils.graph_utils import get_leaf_programs, detect_cycles
        leaves   = set(get_leaf_programs(graph))
        cycles   = {n for c in detect_cycles(graph) for n in c if len(c) > 1}
        for pid, pinfo in kg['programs'].items():
            sm = score_module(pid, pinfo, graph, leaves, cycles)
            assert 0.0 <= sm.risk_score   <= 1.0
            assert 0.0 <= sm.value_score  <= 1.0
            assert 0.0 <= sm.feasibility  <= 1.0
            assert 0.0 <= sm.composite    <= 1.0

    def test_leaf_has_low_risk(self, kg):
        # ACCTPROC is a leaf (no outgoing calls) -> should have risk < INVMGMT
        graph  = kg['_graph']
        from utils.graph_utils import get_leaf_programs, detect_cycles
        leaves = set(get_leaf_programs(graph))
        cycles = {n for c in detect_cycles(graph) for n in c if len(c) > 1}
        sm_acct = score_module('ACCTPROC', kg['programs']['ACCTPROC'],
                               graph, leaves, cycles)
        sm_inv  = score_module('INVMGMT',  kg['programs']['INVMGMT'],
                               graph, leaves, cycles)
        assert sm_acct.is_leaf is True
        assert sm_inv.is_leaf  is False
        assert sm_acct.risk_score <= sm_inv.risk_score

    def test_composite_formula(self, kg):
        graph  = kg['_graph']
        from utils.graph_utils import get_leaf_programs, detect_cycles
        leaves = set(get_leaf_programs(graph))
        cycles = set()
        pid    = list(kg['programs'].keys())[0]
        sm     = score_module(pid, kg['programs'][pid], graph, leaves, cycles)
        expected = round((1.0 - sm.risk_score) * sm.value_score * sm.feasibility, 4)
        assert abs(sm.composite - expected) < 0.001

    def test_cyclic_node_gets_high_risk(self, kg):
        graph  = kg['_graph']
        from utils.graph_utils import get_leaf_programs
        leaves = set(get_leaf_programs(graph))
        # Force ACCTPROC into cyclic set
        cyclic = {'ACCTPROC'}
        sm = score_module('ACCTPROC', kg['programs']['ACCTPROC'],
                          graph, leaves, cyclic)
        assert sm.risk_score >= 0.9
        assert sm.has_cycles is True

# ── Ordering tests ─────────────────────────────────────
class TestOrdering:
    def test_leaf_migrates_before_caller(self, kg):
        migration_plan = plan(kg)
        ids = [m.program_id for m in migration_plan.modules]
        # ACCTPROC (leaf/dependency) must come before INVMGMT (caller)
        assert ids.index('ACCTPROC') < ids.index('INVMGMT')

    def test_priorities_are_sequential(self, kg):
        migration_plan = plan(kg)
        for i, mod in enumerate(migration_plan.modules, 1):
            assert mod.priority == i

    def test_cyclic_modules_at_end(self):
        # Build a mini cyclic kg manually
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(['A', 'B', 'C'],
                         filepath='test.cbl', paragraph_count=2,
                         data_item_count=2, fan_in=1, fan_out=1)
        G.add_edge('A', 'B', type='CALL')
        G.add_edge('B', 'A', type='CALL')   # cycle
        G.add_edge('C', 'B', type='CALL')
        modules = [
            ScoredModule('A', 'A.cbl', has_cycles=True,  composite=0.8),
            ScoredModule('B', 'B.cbl', has_cycles=True,  composite=0.9),
            ScoredModule('C', 'C.cbl', has_cycles=False, composite=0.5),
        ]
        ordered = order_modules(modules, G)
        ids = [m.program_id for m in ordered]
        assert ids.index('C') < ids.index('A')
        assert ids.index('C') < ids.index('B')

# ── Serialisation tests ──────────────────────────────
class TestSerialisation:
    def test_json_has_required_keys(self, kg, tmp_path):
        mp = plan(kg, output_path=str(tmp_path / 'plan.json'))
        data = json.loads((tmp_path / 'plan.json').read_text())
        for key in ['modules', 'cycles', 'total_programs',
                    'leaf_count', 'generated_at']:
            assert key in data

    def test_module_json_has_all_fields(self, kg, tmp_path):
        mp = plan(kg, output_path=str(tmp_path / 'plan.json'))
        data = json.loads((tmp_path / 'plan.json').read_text())
        for mod in data['modules']:
            for field in ['program_id', 'filepath', 'priority',
                          'risk_score', 'value_score', 'feasibility',
                          'composite', 'reason', 'depends_on',
                          'called_by', 'is_leaf', 'has_cycles']:
                assert field in mod, f'Missing field: {field}'

    def test_load_roundtrip(self, kg, tmp_path):
        mp  = plan(kg, output_path=str(tmp_path / 'plan.json'))
        mp2 = load_plan(tmp_path / 'plan.json')
        assert mp2.total_programs == mp.total_programs
        assert mp2.leaf_count     == mp.leaf_count
        ids_orig   = [m.program_id for m in mp.modules]
        ids_loaded = [m.program_id for m in mp2.modules]
        assert ids_orig == ids_loaded

    def test_summary_string(self, kg):
        mp = plan(kg)
        s  = summarise_plan(mp)
        assert 'ACCTPROC' in s
        assert 'INVMGMT'  in s
        assert 'Priority' in s

