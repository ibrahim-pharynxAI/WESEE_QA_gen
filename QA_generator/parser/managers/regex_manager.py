import re

# Regex patterns as compiled objects
RE_JUNK_COMMANDS = re.compile(
    r'(\\Big|\\rceil|\\rfloor|\\rce\w*|\\rc\w*|\\rf\w*|\\r\w*|\\dots|\\quad)\s*'
)

RE_HTML_IMAGE_COMMENT = re.compile(
    r'^\s*<!--\s*image\s*-->\s*$', 
    flags=re.MULTILINE | re.IGNORECASE
)

RE_UNKNOWN_LATEX_COMMAND = re.compile(
    r'\\[a-zA-Z]+'
)

RE_STRAY_BRACES = re.compile(
    r'[\{\}]'
)

RE_WHITESPACE = re.compile(
    r'\s+'
)

RE_LATEX_FORMULAS = re.compile(
    r'(\${1,2})(.*?)(\1)',
    flags=re.DOTALL
)


def remove_junk_commands(latex: str) -> str:
    return RE_JUNK_COMMANDS.sub(' ', latex)


def remove_html_image_comments(latex: str) -> str:
    return RE_HTML_IMAGE_COMMENT.sub('', latex)


def remove_unknown_latex_commands(latex: str) -> str:
    return RE_UNKNOWN_LATEX_COMMAND.sub('', latex)


def remove_stray_braces(latex: str) -> str:
    return RE_STRAY_BRACES.sub('', latex)


def collapse_whitespace(text: str) -> str:
    return RE_WHITESPACE.sub(' ', text)


def clean_edges(text: str) -> str:
    return text.strip(' ,.;')


def clean_latex(latex: str) -> str:
    """
    Apply all regex-based cleaning steps to LaTeX content.
    """
    if not latex or not latex.strip():
        return latex

    latex = remove_junk_commands(latex)
    latex = remove_html_image_comments(latex)
    latex = remove_unknown_latex_commands(latex)
    latex = remove_stray_braces(latex)
    latex = collapse_whitespace(latex)
    latex = clean_edges(latex)

    return latex.strip()


def clean_latex_formulas_in_md(md_text: str) -> str:
    """
    Finds all LaTeX formulas in Markdown text and cleans them in-place.
    """
    def replace_func(match):
        delim = match.group(1)
        latex = match.group(2)
        cleaned = clean_latex(latex)
        return f"{delim}{cleaned}{delim}"

    return RE_LATEX_FORMULAS.sub(replace_func, md_text)

def markdown_splitter(md_text: str) -> str:
    """
    Divide the markdown content into chunks based on '##' subheadings.
    Each chunk is separated by a line of 50 dashes.
    
    Args:
        md_text (str): Markdown text to split.
    
    Returns:
        str: Chunks joined by separator lines.
    """
    # Split on each '##' subheading (but keep the delimiter in result)
    sections = re.split(r'(?=^## )', md_text, flags=re.MULTILINE)

    # Remove empty sections and strip whitespace
    sections = [section.strip() for section in sections if section.strip()]

    # Join with separator
    return ('\n' + '-' * 50 + '\n').join(sections)


if __name__ == "__main__":
    test_md = """
Table 1. The quantitative results of the head reconstruction. 'Portrait' refers to the use of the Portrait-Sync Generator. We achieve state-of-the-art performance on most metrics. We highlight best and second-best results.

| Methods                               |   PSNR ↑ |   LPIPS ↓ |   MS-SSIM ↑ |   FID ↓ |   NIQE ↓ |   BRISQUE ↓ |   LMD ↓ |   AUE ↓ |   LSE-C ↑ |
|---------------------------------------|----------|-----------|-------------|---------|----------|-------------|---------|---------|-----------|
| Wav2Lip (ACM MM20 [34])               |  33.4385 |    0.0697 |      0.9781 | 16.0228 |  14.5367 |     44.2659 |  4.963  |  2.9029 |    9.2387 |
| VideoReTalking (SIGGRAPH Asia 22 [7]) |  31.7923 |    0.0488 |      0.968  |  9.2063 |  14.241  |     43.0465 |  5.8575 |  3.3308 |    7.9683 |
| GAN DINet (AAAI 23 [50])              |  31.6475 |    0.0443 |      0.964  |  9.43   |  14.685  |     40.365  |  4.3725 |  3.6875 |    6.5653 |
| TalkLip (CVPR 23 [41])                |  32.5154 |    0.0782 |      0.9697 | 18.4997 |  14.6385 |     46.6717 |  5.8605 |  2.9579 |    5.9472 |
| IP-LAP (CVPR 23 [53])                 |  35.1525 |    0.0443 |      0.9803 |  8.2125 |  14.64   |     42.075  |  3.335  |  2.84   |    4.9541 |
| AD-NeRF (ICCV 21 [14])                |  26.7291 |    0.1536 |      0.9111 | 28.9862 |  14.9091 |     55.4667 |  2.9995 |  5.5481 |    4.4996 |
| RAD-NeRF (arXiv 22 [38])              |  31.7754 |    0.0778 |      0.9452 |  8.657  |  13.4433 |     44.6892 |  2.9115 |  5.0958 |    5.5219 |
| GeneFace (ICLR 23 [44])               |  24.8165 |    0.1178 |      0.8753 | 21.7084 |  13.3353 |     46.5061 |  4.2859 |  5.4527 |    5.195  |
| NeRF ER-NeRF (ICCV 23 [21])           |  32.5216 |    0.0334 |      0.9501 |  5.2936 |  13.7048 |     34.7361 |  2.8137 |  4.1873 |    5.7749 |
| SyncTalk (w/o Portrait)               |  35.3542 |    0.0235 |      0.9769 |  3.9247 |  13.1333 |     33.2954 |  2.5714 |  2.5796 |    8.1331 |
| SyncTalk (Portrait)                   |  37.4017 |    0.0113 |      0.9841 |  2.707  |  14.2165 |     37.3042 |  2.5043 |  3.2074 |    8.0263 |

Portrait-Sync Generator. During the training process, to address NeRF's limitations in capturing fine details like hair strands and dynamic backgrounds, we introduce a PortraitSync Generator with two key sections. First, NeRF renders the face area ( F r ), creates G ( F r ) through Gaussian blur, and then uses our synchronized head pose to be able to merge with the original image ( F o ) to enhance hair detail fidelity. Second, when the head and torso are combined, if the character in the source video speaks while the generated face is silent, a dark gap area might appear, as shown in Fig. 5 ( b ) . We fill these areas with the average neck color ( C n ). This approach results in more realistic details and improved visual quality through the Portrait-Sync Generator.

# 4. Experiments

## 4.1. Experimental Settings

Dataset. To ensure a fair comparison, we use the same welledited video sequences from [14, 21, 44], including English and French. The average length of these videos is approximately 8,843 frames, and each video is recorded at 25 FPS. Except for the video from AD-NeRF [14], which has a resolution of 450 × 450 , all other videos have a resolution of 512 × 512 , with the character-centered.

Comparison Baselines. We compare our method with five GAN-based methods, including Wav2Lip [34], VideoReTalking [7], DINet [50], TalkLip [41], and IP-LAP [53], and NeRF-based methods such as AD-NeRF [14], RADNeRF [38], GeneFace [44], and ER-NeRF [21].

Implementation Details. In the coarse stage, the portrait head is trained for 100,000 iterations and 25,000 in the fine stage, sampling 256 2 rays per iteration using a 2D hash encoder ( L =14, F =1). We employ the AdamW optimizer [23], with learning rates of 0.01 for the hash encoder and 0.001 for other modules. Total training time is approximately 2 hours on an NVIDIA RTX 3090 GPU.

Table 2. The quantitative results of the lip synchronization. We use two different audio samples to drive the same subject, then highlight best and second-best results.

|                         | Audio A   | Audio A   | Audio B   | Audio B   |
|-------------------------|-----------|-----------|-----------|-----------|
| Methods                 | LSE-D ↓   | LSE-C ↑   | LSE-D ↓   | LSE-C ↑   |
| DINet (AAAI 23 [50])    | 8.5031    | 5.6956    | 8.2038    | 5.1134    |
| TalkLip (CVPR 23 [41])  | 8.7615    | 5.7449    | 8.7019    | 5.5359    |
| IP-LAP (CVPR 23 [53])   | 9.8037    | 3.8578    | 9.1102    | 4.389     |
| GeneFace (ICLR 23 [44]) | 9.5451    | 4.2933    | 9.6675    | 3.7342    |
| ER-NeRF (ICCV 23 [21])  | 11.813    | 2.4076    | 10.7338   | 3.0242    |
| SyncTalk (Ours)         | 7.7211    | 6.6659    | 8.0248    | 6.2596    |

## 4.2. Quantitative Evaluation

Full Reference Quality Assessment. In terms of image quality, we use full reference metrics such as Peak Signalto-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS) [46], Multi-Scale Structure Similarity (MS-SSIM), and Frechet Inception Distance (FID) [15] as evaluation metrics.

No Reference Quality Assessment. In high PSNR images, texture details may not align with human visual perception [47]. For more precise output definition and comparison, we use two No Reference methods: the Natural Image Quality Evaluator (NIQE)[29] and the Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)[28].

Synchronization Assessment. For synchronization, we use landmark distance (LMD) to measure the synchronicity of facial movements, action units error (AUE) [4] to assess the accuracy of facial movements, and introduce Lip Sync Error Confidence (LSE-C), consistent with Wav2Lip [34], to evaluate the synchronization between lip movements and audio. Evaluation Results. The evaluation results of the head reconstruction are shown in Tab. 1. We compare recent methods based on GAN and NeRF. It can be observed that our image quality is superior to other methods in all aspects. In terms of synchronization, our results surpass most methods.
"""

    cleaned = clean_latex_formulas_in_md(test_md)
    print(markdown_splitter(cleaned))
