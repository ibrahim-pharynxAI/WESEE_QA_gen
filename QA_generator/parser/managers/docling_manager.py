from pathlib import Path
import time

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.exceptions import ConversionError

from .regex_manager import clean_latex, clean_latex_formulas_in_md
from .log_manager import LogManager


class PDFConverter:
    def __init__(
        self,
        cache_dir="QA_generator/parser/TempParsed",
        output_dir="QA_generator/parser/parsed_outputs",
        do_ocr=True,
        do_formula_enrichment=True,
        do_table_structure=True,
        do_figure_enrichment=True,
    ):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.logger = LogManager().get_logger()
        self.pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_formula_enrichment=do_formula_enrichment,
            do_figure_enrichment=do_figure_enrichment,
            do_ocr=do_ocr,
        )

    def clean_latex(self, latex):
        return clean_latex(latex)

    def clean_latex_formulas_in_md(self, md_text):
        return clean_latex_formulas_in_md(md_text)

    def convert_pdf(self, pdf_path):
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

        pdf_name = pdf_path.stem
        output_md_filename = self.output_dir / f"{pdf_name}_parsed.md"
        uncleaned_md_path = self.cache_dir / f"{pdf_name}_uncleaned.md"

        if output_md_filename.is_dir():
            raise IsADirectoryError(f"'{output_md_filename}' is a directory.")

        self.logger.info(f"Starting conversion of {pdf_path}")

        format_opts = PdfFormatOption(pipeline_options=self.pipeline_options)
        converter = DocumentConverter(format_options={InputFormat.PDF: format_opts})

        try:
            start_time = time.perf_counter()
            self.logger.info("Converting PDF to Markdown")
            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()

            end_time = time.perf_counter()

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            uncleaned_md_path.write_text(markdown_text, encoding="utf-8")
            self.logger.info(f"Saved original Markdown to: {uncleaned_md_path}")

            self.logger.info("Cleaning LaTeX formulas")
            cleaned_md = self.clean_latex_formulas_in_md(markdown_text)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_md_filename.write_text(cleaned_md, encoding="utf-8")

            self.logger.info(f"Saved cleaned Markdown to: {output_md_filename}")
            self.logger.info(f"Time taken: {end_time - start_time:.2f} seconds")

            return {
                "output_md_path": str(output_md_filename),
                "time_elapsed": end_time - start_time,
            }

        except ConversionError as e:
            self.logger.error(f"Docling conversion failed: {str(e)}")
            raise
