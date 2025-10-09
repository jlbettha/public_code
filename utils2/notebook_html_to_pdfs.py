import os
import asyncio
from playwright.async_api import async_playwright


async def html_to_pdf(html_content, output_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content)
        await page.pdf(
            path=output_path,
            format="Legal",
            scale=0.6,
            margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"},
            header_template='<span style="font-size:12px">Header Content</span>',
            footer_template='<span style="font-size:12px">Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>',
            display_header_footer=True,
        )
        await browser.close()


def main():
    pwd = os.getcwd()
    html_directory = os.path.join(pwd, "saved_notebook_docs")  # Directory containing Jupyter notebooks
    for dir, subdirs, files in os.walk(html_directory):
        print(f"Current directory: {dir}")
        print(f"Subdirectories: {subdirs}")
        print(f"Files: {files}")

        for file in files:
            if file.endswith(".html"):
                html_file_path = os.path.join(dir, file)
                pdf_file_path = os.path.join(dir, file.replace(".html", ".pdf"))
                print(f"Converting {html_file_path} to {pdf_file_path}")
                with open(html_file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                try:
                    asyncio.run(html_to_pdf(html_content, pdf_file_path))
                except Exception as e:
                    msg = f"Error converting {file} to PDF: {e}"
                    print(msg)
                    exit(1)


if __name__ == "__main__":
    main()
