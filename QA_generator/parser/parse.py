from .managers.docling_manager import PDFConverter
from .managers.log_manager import LogManager

# Initialize logging
logger = LogManager().get_logger()


def main(pdf_path: str, cache_dir: str = "QA_generator/parser/TempParsed"):
    try:
        converter = PDFConverter(cache_dir=cache_dir)
        result = converter.convert_pdf(pdf_path)

        logger.success("Conversion completed successfully!")
        logger.info(f"Output file: {result['output_md_path']}")
        logger.info(f"Processing time: {result['time_elapsed']:.2f} seconds")

        return result
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
