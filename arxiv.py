import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json

DATA_DIR = Path('./data')

@dataclass
class ArxivPaper:
    """Data class to store paper information"""
    title: str
    summary: str
    html_link: str
    authors: List[str]
    published_date: str
    updated_date: str
    arxiv_id: str
    search_term: str

class ArxivFetcher:
    """Class to fetch and parse papers from arXiv API"""
    
    BASE_URL = "http://export.arxiv.org/api/query?"
    NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    def __init__(self):
        self.last_query_url = None

    def fetch_papers(self, 
                    search_query: str,
                    max_results: int = 10,
                    sort_by: str = "lastUpdatedDate",
                    sort_order: str = "descending") -> List[ArxivPaper]:
        """Fetch papers from arXiv based on search parameters"""
        try:
            time.sleep(3)  # Respect rate limits
            
            params = {
                "search_query": search_query,
                "sortBy": sort_by,
                "sortOrder": sort_order,
                "start": 40,
                "max_results": str(max_results)
            }
            
            query_string = urllib.parse.urlencode(params)
            self.last_query_url = self.BASE_URL + query_string
            
            with urllib.request.urlopen(self.last_query_url) as response:
                xml_data = response.read().decode('utf-8')
            
            return self._parse_xml(xml_data, search_query)
            
        except Exception as e:
            print(f"Error fetching data from arXiv: {e}")
            return []

    def _parse_xml(self, xml_data: str, search_term: str) -> List[ArxivPaper]:
        """Parse XML response from arXiv API"""
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall('atom:entry', self.NAMESPACES):
            try:
                title = entry.find('atom:title', self.NAMESPACES).text.strip()
                summary = entry.find('atom:summary', self.NAMESPACES).text.strip()
                
                # Get HTML link instead of PDF
                arxiv_id = entry.find('atom:id', self.NAMESPACES).text.split('/')[-1]
                html_link = f"https://arxiv.org/html/{arxiv_id}"
                
                authors = [author.find('atom:name', self.NAMESPACES).text
                          for author in entry.findall('atom:author', self.NAMESPACES)]
                
                published = entry.find('atom:published', self.NAMESPACES).text
                updated = entry.find('atom:updated', self.NAMESPACES).text

                papers.append(ArxivPaper(
                    title=title,
                    summary=summary,
                    html_link=html_link,
                    authors=authors,
                    published_date=published,
                    updated_date=updated,
                    arxiv_id=arxiv_id,
                    search_term=search_term
                ))
                
            except AttributeError as e:
                print(f"Error parsing paper entry: {e}")
                continue
                
        return papers

class HTMLHandler:
    """Class to handle HTML downloads and content extraction"""
    
    def __init__(self, cache_dir: str = "paper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def download_and_parse_html(self, html_url: str, arxiv_id: str) -> Tuple[bool, str]:
        """Download and extract content from HTML page"""
        try:
            # Check cache first
            cache_path = self.cache_dir / f"{arxiv_id}.html"
            if cache_path.exists():
                return self._parse_cached_html(cache_path)
            
            # Download the HTML
            print(f"Downloading HTML for {arxiv_id}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            request = urllib.request.Request(html_url, headers=headers)
            
            with urllib.request.urlopen(request) as response:
                html_content = response.read().decode('utf-8')
            
            # Save to cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return self._parse_html_content(html_content)
            
        except Exception as e:
            return False, f"Error processing HTML: {str(e)}"

    def _parse_html_content(self, html_content: str) -> Tuple[bool, str]:
        """Parse HTML content using BeautifulSoup"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            content_parts = []
            # Extract main content - adjust selectors based on arXiv's HTML structure
            abstract = soup.find('blockquote', class_='abstract')
            
            # Add SECTION prefix to headings and content
            if abstract:
                content_parts.append("SECTION: Abstract\n" + abstract.get_text(strip=True) + "\n\n")
            
            # Remove math tags
            math_tags = soup.find_all('math')
            for math in math_tags:
                math.decompose()  # Using decompose()
            
            # Track heading levels to create a more structured output
            current_section = None
            for section in soup.find_all(['h1', 'h2', 'p', 'h3']):
                if section.name in ['h1', 'h2', 'h3']:
                    # Start a new section when a heading is encountered
                    current_section = f"SECTION: {section.get_text(strip=True)}"
                    content_parts.append(current_section + "\n")
                elif section.name == 'p' and current_section is not None:
                    # Add paragraph text to the current section
                    section_text = section.get_text(strip=True)
                    if section_text:  # Only add non-empty sections
                        content_parts.append(section_text + "\n")
            
            main_content = "\n".join(content_parts)
            return True, main_content.strip()
            
        except Exception as e:
            return False, f"Error parsing HTML content: {str(e)}"
    def _parse_cached_html(self, file_path: Path) -> Tuple[bool, str]:
        """Parse cached HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return self._parse_html_content(html_content)
                
        except Exception as e:
            return False, f"Error reading cached HTML: {str(e)}"

    def clear_cache(self):
        """Clear the HTML cache directory"""
        for file in self.cache_dir.glob("*.html"):
            file.unlink()

def process_papers_for_term(
    search_term: str,
    max_results: int,
    html_handler: HTMLHandler,
    download_html: bool = True,
    save_text: bool = True,
    text_dir: str = "paper_texts",
    sort_by: str = "lastUpdatedDate",
    sort_order: str = "descending"
) -> List[dict]:
    """Process papers for a single search term"""
    fetcher = ArxivFetcher()
    papers = fetcher.fetch_papers(
        search_query=search_term,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    # Create text directory if it doesn't exist
    text_path = DATA_DIR / text_dir
    text_path.mkdir(exist_ok=True)
    
    results = []
    
    for paper in papers:
        paper_info = {
            'title': paper.title,
            'authors': paper.authors,
            'published_date': paper.published_date,
            'summary': paper.summary,
            'arxiv_id': paper.arxiv_id,
            'html_link': paper.html_link,
            'search_term': paper.search_term
        }
        
        if download_html:
            success, content = html_handler.download_and_parse_html(
                paper.html_link,
                paper.arxiv_id
            )
            
            if success:
                paper_info['html_content'] = content
                
                if save_text:
                    filename = text_path / f"{paper.arxiv_id}_content.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    paper_info['text_file'] = str(filename)
            else:
                paper_info['html_error'] = content
        
        results.append(paper_info)
    
    return results

def process_multiple_search_terms(
    search_terms: List[str],
    papers_per_term: int = 10,
    download_html: bool = True,
    save_text: bool = True,
    text_dir: str = "paper_texts",
    cache_dir: str = "paper_cache",
    sort_by: str = "lastUpdatedDate",
    sort_order: str = "descending",
    max_workers: int = 3
) -> Dict[str, List[dict]]:
    """Process multiple search terms in parallel"""
    html_handler = HTMLHandler(cache_dir=cache_dir)
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_term = {
            executor.submit(
                process_papers_for_term,
                term,
                papers_per_term,
                html_handler,
                download_html,
                save_text,
                text_dir,
                sort_by,
                sort_order
            ): term for term in search_terms
        }
        
        for future in as_completed(future_to_term):
            term = future_to_term[future]
            try:
                results[term] = future.result()
            except Exception as e:
                print(f"Error processing search term '{term}': {e}")
                results[term] = []
    
    return results

if __name__ == "__main__":
    # Example usage
    search_terms = [
        'ti:"AI"',
        'ti:"Unsupervised Learning"',
        'ti:"Random Forest"',
        'ti:"Supervised Learning"',
    ]
    
    results = process_multiple_search_terms(
        search_terms=search_terms,
        papers_per_term=20,
        download_html=True,
        save_text=True,
        cache_dir="paper_cache"
    )

    with open(DATA_DIR/'paper_info.json', 'r') as f:
        info = json.load(f)

    with open(DATA_DIR/'paper_info.json', 'w') as f:
        info.extend(results.items())
        json.dump(info, f)

    # Print results summary
    for term, papers in results.items():
        print(f"\n=== Results for {term} ===")
        print(f"Found {len(papers)} papers")

        for paper in papers:
            print("\n" + "="*80)
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Published: {paper['published_date']}")
            print(f"HTML Link: {paper['html_link']}")
            print(f"Summary: {paper['summary'][:200]}...")
            if 'html_content' in paper:
                print(f"HTML Content Length: {len(paper['html_content'])} characters")
                if 'text_file' in paper:
                    print(f"Text saved to: {paper['text_file']}")
            elif 'html_error' in paper:
                print(f"HTML Error: {paper['html_error']}")