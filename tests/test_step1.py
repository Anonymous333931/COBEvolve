# ── FILE: cobol_moderniser/tests/test_step1.py ────────────────────
"""
test_step1.py -- Verify the Step 1 foundation components.


Run: pytest tests/test_step1.py -v
"""


import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
import time
from pathlib import Path
from utils.cobol_parser import parse_cobol_source, parse_cobol_file
from core.knowledge_base import KnowledgeBase, TranslationRecord, FailureRecord
from config import config




SAMPLE_COBOL = '''
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TESTPROG.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-NAME         PIC X(20).
       01 WS-COUNT        PIC 9(5) VALUE 0.
       01 WS-AMOUNT       PIC 9(7)V99 VALUE 0.
       PROCEDURE DIVISION.
       MAIN-PARA.
           MOVE 'HELLO' TO WS-NAME.
           ADD 1 TO WS-COUNT.
           DISPLAY WS-NAME.
           PERFORM CALC-PARA.
           STOP RUN.
       CALC-PARA.
           COMPUTE WS-AMOUNT = WS-COUNT * 100.
           DISPLAY WS-AMOUNT.
'''




class TestCOBOLParser:
    def test_parse_program_id(self):
        prog = parse_cobol_source(SAMPLE_COBOL)
        assert prog.program_id == 'TESTPROG'


    def test_parse_data_items(self):
        prog = parse_cobol_source(SAMPLE_COBOL)
        assert len(prog.working_storage) >= 3
        names = [item.name for item in prog.all_data_items]
        assert 'WS-NAME' in names
        assert 'WS-COUNT' in names


    def test_parse_paragraphs(self):
        prog = parse_cobol_source(SAMPLE_COBOL)
        assert len(prog.paragraphs) >= 2
        assert 'MAIN-PARA' in prog.paragraph_names
        assert 'CALC-PARA' in prog.paragraph_names

    def test_parse_numeric_paragraph_headers(self):
        src = '''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. NUMPARA.
        PROCEDURE DIVISION.
        0001-MAIN.
            PERFORM 0400-WORK.
            STOP RUN.
        0400-WORK.
            DISPLAY 'OK'.
        '''
        prog = parse_cobol_source(src)
        assert '0001-MAIN' in prog.paragraph_names
        assert '0400-WORK' in prog.paragraph_names

    def test_split_display_and_accept_into_distinct_statements(self):
        src = '''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. ACCEPTSPLIT.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        77 WRK-NOME PIC X(10).
        PROCEDURE DIVISION.
        MAIN-PROCEDURE.
            DISPLAY 'NOME: '
            ACCEPT WRK-NOME.
            STOP RUN.
        '''
        prog = parse_cobol_source(src)
        main = prog.paragraphs[0]
        assert [stmt.verb for stmt in main.statements] == ['DISPLAY', 'ACCEPT', 'STOP']

    def test_same_line_display_and_accept_are_split(self):
        src = '''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. INLINEACCEPT.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        77 WRK-NOME PIC X(10).
        PROCEDURE DIVISION.
        MAIN-PROCEDURE.
            DISPLAY 'NOME:' . ACCEPT WRK-NOME.
            STOP RUN.
        '''
        prog = parse_cobol_source(src)
        main = prog.paragraphs[0]
        assert [stmt.verb for stmt in main.statements] == ['DISPLAY', 'ACCEPT', 'STOP']

    def test_indented_display_operand_is_not_treated_as_paragraph(self):
        src = '''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. NOPARABUG.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 WS-REC.
           05 WS-NOME PIC X(10).
           05 WS-SALARIO PIC 9(08).
        PROCEDURE DIVISION.
            MAIN-PARA.
                DISPLAY 'CAD: '
                    WS-NOME
                    ' '
                    WS-SALARIO.
                STOP RUN.
        '''
        prog = parse_cobol_source(src)
        assert prog.paragraph_names == ['MAIN-PARA']

    def test_parse_occurs_without_times_keyword(self):
        src = '''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. OCCURSNO.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 WS-TABLE.
           03 WS-ENTRY OCCURS 2.
              05 WS-NAME PIC X(03).
        PROCEDURE DIVISION.
            STOP RUN.
        '''
        prog = parse_cobol_source(src)
        table = prog.working_storage[0]
        assert table.children[0].occurs == 2

    def test_parse_multiline_value_preserves_fixed_width_spaces(self):
        src = '''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. WRAPVAL.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 WS-DATA.
           03 FILLER PIC X(12) VALUE
               "ABCD    1234".
        PROCEDURE DIVISION.
            STOP RUN.
        '''
        prog = parse_cobol_source(src)
        filler = prog.working_storage[0].children[0]
        assert filler.value == '"ABCD    1234"'

    def test_indented_free_format_data_items_do_not_swallow_following_fields(self):
        src = '''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. FREEFMT.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
            01  WS-ENV-LEN5                 PIC 9(4) COMP-5.
            01  WS-DATADIR                 PIC X(32) VALUE "${COBCURSES_DATADIR}".
            01  WS-FLAG                    PIC X.
                88  WS-ON                  VALUE 'Y' FALSE IS 'N'.
            01  WS-NEXT                    PIC X VALUE 'N'.
        PROCEDURE DIVISION.
            STOP RUN.
        '''
        prog = parse_cobol_source(src)
        names = [item.name for item in prog.working_storage]
        assert names == ['WS-ENV-LEN5', 'WS-DATADIR', 'WS-FLAG', 'WS-NEXT']
        ws_on = prog.working_storage[2].children[0]
        assert ws_on.value == "'Y' FALSE IS 'N'"

    @pytest.mark.timeout(3)
    def test_long_if_chain_parses_without_quadratic_slowdown(self):
        repeated_ifs = '\n'.join(
            f'            IF WS-COUNT = {idx}\n'
            "                PERFORM GOOD ELSE PERFORM BAD."
            for idx in range(1, 181)
        )
        src = f'''
        IDENTIFICATION DIVISION.
        PROGRAM-ID. LONGIF.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        77 WS-COUNT PIC 9(3) VALUE 1.
        77 GOOD-FLAG PIC 9 VALUE 0.
        PROCEDURE DIVISION.
        MAIN-PARA.
{repeated_ifs}
            STOP RUN.
        GOOD.
            MOVE 1 TO GOOD-FLAG.
        BAD.
            MOVE 0 TO GOOD-FLAG.
        '''
        started = time.time()
        prog = parse_cobol_source(src)
        elapsed = time.time() - started
        main = next(p for p in prog.paragraphs if p.name == 'MAIN-PARA')
        assert len(main.statements) == 181
        assert elapsed < 1.5


    def test_perform_tracking(self):
        prog = parse_cobol_source(SAMPLE_COBOL)
        main = next(p for p in prog.paragraphs if p.name == 'MAIN-PARA')
        assert 'CALC-PARA' in main.performs


    def test_data_item_python_type(self):
        prog = parse_cobol_source(SAMPLE_COBOL)
        ws_name = next(i for i in prog.all_data_items if i.name == 'WS-NAME')
        ws_amount = next(i for i in prog.all_data_items if i.name == 'WS-AMOUNT')
        assert ws_name.python_type == 'str'
        assert ws_amount.python_type == 'Decimal'


    def test_python_name_conversion(self):
        prog = parse_cobol_source(SAMPLE_COBOL)
        ws_count = next(i for i in prog.all_data_items if i.name == 'WS-COUNT')
        assert ws_count.python_name == 'ws_count'




class TestKnowledgeBase:
    @pytest.fixture
    def kb(self, tmp_path):
        return KnowledgeBase(
            db_path=str(tmp_path / 'test.db'),
            chroma_path=str(tmp_path / 'chroma'))


    def test_save_and_lookup_translation(self, kb):
        cobol = 'MOVE 1 TO WS-COUNT.'
        rec = TranslationRecord(
            cobol_hash=kb.hash_cobol(cobol),
            program_id='TESTPROG',
            cobol_code=cobol,
            translated_code='ws_count = 1',
            language='python',
            success=True,
            accuracy_score=1.0)
        kb.save_translation(rec)
        found = kb.lookup_translation(cobol)
        assert found is not None
        assert found.program_id == 'TESTPROG'
        assert found.translated_code == 'ws_count = 1'


    def test_cache_miss_returns_none(self, kb):
        result = kb.lookup_translation('MOVE 99 TO WS-X.')
        assert result is None


    def test_save_failure(self, kb):
        rec = FailureRecord(
            program_id='TESTPROG',
            cobol_code='DIVIDE 0 INTO WS-X.',
            translated_code='ws_x /= 0',
            error_message='ZeroDivisionError',
            diagnosis='Division by zero not handled',
            fix_applied='Add zero check before division',
            resolved=True)
        row_id = kb.save_failure(rec)
        assert row_id > 0


    def test_stats(self, kb):
        stats = kb.stats()
        assert 'total_translations' in stats
        assert 'chroma_indexed_failures' in stats




class TestConfig:
    def test_ollama_url_set(self):
        assert config.OLLAMA_BASE_URL.startswith('http')


    def test_model_names_set(self):
        assert 'codellama' in config.MODEL_TRANSLATION or 'llama' in config.MODEL_TRANSLATION


    def test_setup_creates_dirs(self, tmp_path):
        import os
        os.environ['COBOL_OUTPUT_DIR'] = str(tmp_path / 'output')
        from config import Config
        c = Config().setup()
        assert (tmp_path / 'output').exists()




class TestOllamaClient:
    def test_import(self):
        from utils.ollama_client import OllamaClient
        client = OllamaClient()
        assert client.base_url.startswith('http')


    @pytest.mark.timeout(10)
    def test_ollama_reachable(self):
        from config import config
        reachable = config.verify_ollama()
        if not reachable:
            pytest.skip('Ollama server not running — start with: ollama serve')
