# ── FILE: cobol_moderniser/tests/test_step2.py ────────────────────
"""
test_step2.py -- Tests for the Comprehension Agent.


Run: pytest tests/test_step2.py -v


Tests that require Ollama are auto-skipped if Ollama is not running.
"""


import sys, os, json, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from pathlib import Path
from utils.cobol_parser import parse_cobol_source
from utils.graph_utils import (
    build_dependency_graph, get_leaf_programs,
    detect_cycles, topological_order, graph_summary
)
from agents.comprehension_agent import (
    scan_cobol_files, build_graph, extract_business_rules,
    assemble_knowledge_graph, comprehend, summarise_knowledge_graph
)
from config import config




# ── Shared test fixtures ─────────────────────────────────────────
ACCTPROC_SRC = '''
       IDENTIFICATION DIVISION.
       PROGRAM-ID. ACCTPROC.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-BALANCE        PIC 9(8)V99 VALUE 0.
       01 WS-MIN-BALANCE    PIC 9(7)V99 VALUE 100.00.
       01 WS-MAX-TRANSACTION PIC 9(7)V99 VALUE 50000.00.
       01 WS-RESULT-CODE    PIC 9(2) VALUE 0.
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
def two_programs():
    return [
        parse_cobol_source(ACCTPROC_SRC, 'ACCTPROC.cbl'),
        parse_cobol_source(INVMGMT_SRC,  'INVMGMT.cbl'),
    ]




def ollama_available() -> bool:
    return config.verify_ollama()




# ── Phase 1: Scan ─────────────────────────────────────────────────
class TestPhase1Scan:
    def test_scan_single_program(self, two_programs):
        # scan_cobol_files accepts pre-parsed list (pass-through)
        programs = scan_cobol_files(two_programs)
        assert len(programs) == 2
        ids = [p.program_id for p in programs]
        assert 'ACCTPROC' in ids
        assert 'INVMGMT' in ids


    def test_scan_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            scan_cobol_files('/nonexistent/path')


    def test_scan_directory(self, tmp_path):
        (tmp_path / 'prog1.cbl').write_text(ACCTPROC_SRC)
        (tmp_path / 'prog2.cbl').write_text(INVMGMT_SRC)
        programs = scan_cobol_files(tmp_path)
        assert len(programs) == 2




# ── Phase 2: Graph ────────────────────────────────────────────────
class TestPhase2Graph:
    def test_graph_has_correct_nodes(self, two_programs):
        G = build_dependency_graph(two_programs)
        assert 'ACCTPROC' in G.nodes
        assert 'INVMGMT' in G.nodes


    def test_call_creates_edge(self, two_programs):
        G = build_dependency_graph(two_programs)
        # INVMGMT calls ACCTPROC -> edge must exist
        assert G.has_edge('INVMGMT', 'ACCTPROC')


    def test_fan_in_fan_out(self, two_programs):
        G = build_dependency_graph(two_programs)
        # ACCTPROC is called by INVMGMT -> fan_in=1
        assert G.nodes['ACCTPROC']['fan_in'] == 1
        # INVMGMT calls ACCTPROC -> fan_out=1
        assert G.nodes['INVMGMT']['fan_out'] == 1


    def test_leaf_detection(self, two_programs):
        G = build_dependency_graph(two_programs)
        leaves = get_leaf_programs(G)
        # ACCTPROC has no outgoing inter-program edges -> leaf
        assert 'ACCTPROC' in leaves
        # INVMGMT calls ACCTPROC -> not a leaf
        assert 'INVMGMT' not in leaves


    def test_no_cycles(self, two_programs):
        G = build_dependency_graph(two_programs)
        cycles = detect_cycles(G)
        # Filter out self-loops
        real_cycles = [c for c in cycles if len(c) > 1]
        assert len(real_cycles) == 0


    def test_topological_order(self, two_programs):
        G = build_dependency_graph(two_programs)
        order = topological_order(G)
        assert 'ACCTPROC' in order
        assert 'INVMGMT' in order
        # ACCTPROC (leaf/dependency) should come before INVMGMT (caller)
        if 'ACCTPROC' in order and 'INVMGMT' in order:
            assert order.index('ACCTPROC') < order.index('INVMGMT')


    def test_graph_summary_string(self, two_programs):
        G = build_dependency_graph(two_programs)
        summary = graph_summary(G)
        assert 'programs' in summary
        assert 'edges' in summary




# ── Phase 3: Business Rules ───────────────────────────────────────
class TestPhase3Rules:
    def test_extraction_returns_list(self, two_programs):
        prog = two_programs[0]  # ACCTPROC
        if not ollama_available():
            pytest.skip('Ollama not running')
        rules = extract_business_rules(prog)
        assert isinstance(rules, list)


    def test_rules_have_required_fields(self, two_programs):
        prog = two_programs[0]
        if not ollama_available():
            pytest.skip('Ollama not running')
        rules = extract_business_rules(prog)
        if rules:  # May be empty if Ollama found nothing
            for r in rules:
                assert 'rule' in r
                assert 'paragraph' in r
                assert 'confidence' in r
                assert 0.0 <= r['confidence'] <= 1.0


    def test_extraction_fallback_on_empty_program(self):
        empty_prog = parse_cobol_source('''
               IDENTIFICATION DIVISION.
               PROGRAM-ID. EMPTY.
        ''')
        rules = extract_business_rules(empty_prog)
        assert rules == []  # No paragraphs -> no rules




# ── Phase 4: Knowledge Graph ─────────────────────────────────────
class TestPhase4KnowledgeGraph:
    def test_kg_structure(self, two_programs):
        G = build_dependency_graph(two_programs)
        kg = assemble_knowledge_graph(
            two_programs, G, extract_rules=False)
        assert 'programs' in kg
        assert 'edges' in kg
        assert '_graph' in kg


    def test_kg_has_all_programs(self, two_programs):
        G = build_dependency_graph(two_programs)
        kg = assemble_knowledge_graph(two_programs, G, extract_rules=False)
        assert 'ACCTPROC' in kg['programs']
        assert 'INVMGMT' in kg['programs']


    def test_kg_program_fields(self, two_programs):
        G = build_dependency_graph(two_programs)
        kg = assemble_knowledge_graph(two_programs, G, extract_rules=False)
        prog = kg['programs']['ACCTPROC']
        for field in ['filepath','paragraphs','data_items',
                      'business_rules','calls_external','fan_in','fan_out']:
            assert field in prog, f'Missing field: {field}'


    def test_kg_edges_recorded(self, two_programs):
        G = build_dependency_graph(two_programs)
        kg = assemble_knowledge_graph(two_programs, G, extract_rules=False)
        call_edges = [e for e in kg['edges'] if e['type'] == 'CALL']
        assert len(call_edges) >= 1
        assert any(e['from'] == 'INVMGMT' and e['to'] == 'ACCTPROC'
                   for e in call_edges)


    def test_full_comprehend_no_ollama(self, tmp_path):
        (tmp_path / 'ACCTPROC.cbl').write_text(ACCTPROC_SRC)
        (tmp_path / 'INVMGMT.cbl').write_text(INVMGMT_SRC)
        kg = comprehend(tmp_path, extract_rules=False)
        assert len(kg['programs']) == 2
        summary = summarise_knowledge_graph(kg)
        assert 'Programs: 2' in summary


    @pytest.mark.timeout(120)
    def test_full_comprehend_with_ollama(self, tmp_path):
        if not ollama_available():
            pytest.skip('Ollama not running')
        (tmp_path / 'ACCTPROC.cbl').write_text(ACCTPROC_SRC)
        kg = comprehend(tmp_path, extract_rules=True)
        acct = kg['programs'].get('ACCTPROC', {})
        assert isinstance(acct.get('business_rules', []), list)
