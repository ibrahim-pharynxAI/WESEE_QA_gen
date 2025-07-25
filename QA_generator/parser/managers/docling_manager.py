from pathlib import Path
import time
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

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
        images_dir="QA_generator/parser/images",
        do_ocr=True,
        do_formula_enrichment=True,
        do_table_structure=True,
        do_figure_enrichment=True,
        images_scale=5.0
    ):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.logger = LogManager().get_logger()
        self.pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_formula_enrichment=do_formula_enrichment,
            do_figure_enrichment=do_figure_enrichment,
            do_ocr=do_ocr,
            images_scale=images_scale,
            generate_page_images=True,
            generate_picture_images=True
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
        images_output_dir = self.images_dir / pdf_name

        if output_md_filename.is_dir():
            raise IsADirectoryError(f"'{output_md_filename}' is a directory.")

        self.logger.info(f"Starting conversion of {pdf_path}")

        format_opts = PdfFormatOption(pipeline_options=self.pipeline_options)
        converter = DocumentConverter(format_options={InputFormat.PDF: format_opts})

        try:
            start_time = time.perf_counter()
            self.logger.info("Converting PDF to Markdown")
            result = converter.convert(str(pdf_path))

            # Create images directory for this PDF
            images_output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created images directory: {images_output_dir}")

            # Extract and save images from figures and tables
            table_counter = 0
            picture_counter = 0
            
            for element, _level in result.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = images_output_dir / f"{pdf_name}-table-{table_counter}.png"
                    with element_image_filename.open("wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")
                    self.logger.info(f"Saved table image: {element_image_filename}")

                if isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = images_output_dir / f"{pdf_name}-picture-{picture_counter}.png"
                    with element_image_filename.open("wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")
                    self.logger.info(f"Saved picture image: {element_image_filename}")

            # Instead of using export_to_markdown, we'll build the markdown content manually
            # with proper image references pointing to our dedicated images folder
            markdown_parts = []
            
            for element, level in result.document.iterate_items():
                if hasattr(element, 'export_to_markdown'):
                    # For PictureItem and TableItem, we need to pass the document
                    if isinstance(element, (PictureItem, TableItem)):
                        md_content = element.export_to_markdown(result.document)
                    else:
                        md_content = element.export_to_markdown()
                    
                    # Replace image references to point to our dedicated folder
                    if isinstance(element, PictureItem):
                        picture_counter += 1
                        img_ref = f"../images/{pdf_name}/{pdf_name}-picture-{picture_counter}.png"
                        md_content = md_content.replace('![](images/', f'![]({img_ref})')
                        md_content = md_content.replace('![](images', f'![]({img_ref})')
                    elif isinstance(element, TableItem):
                        table_counter += 1
                        img_ref = f"../images/{pdf_name}/{pdf_name}-table-{table_counter}.png"
                        md_content = md_content.replace('![](tables/', f'![]({img_ref})')
                        md_content = md_content.replace('![](tables', f'![]({img_ref})')
                    
                    markdown_parts.append(md_content)
            
            markdown_text = '\n\n'.join(markdown_parts)

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
                "images_dir": str(images_output_dir),
                "time_elapsed": end_time - start_time,
                "image_count": {
                    "tables": table_counter,
                    "pictures": picture_counter
                }
            }

        except ConversionError as e:
            self.logger.error(f"Docling conversion failed: {str(e)}")
            raise
