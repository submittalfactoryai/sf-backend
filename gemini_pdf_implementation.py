from google import genai
from google.genai import types
import pathlib
import httpx
import argparse
import json
import sys

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract products from construction specification PDF')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: output.json)', default='output.json')
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    filepath = pathlib.Path(args.pdf_path)
    if not filepath.exists():
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        sys.exit(1)
    
    client = genai.Client(api_key="AIzaSyCVCccy3bgPrOU565nfKipkglAHQKqovtg")

    prompt = """
### **INSTRUCTION**

From the provided construction specification text, extract all distinct products into a single JSON array. A product is a specific, orderable item, uniquely identified by its name combined with critical technical attributes like type, grade, size, or model number.

Recheck if all the distinct products are extracted. Manufacturers and preferred products in their catalogs are not to be considered as products.

---

### **Core Principle: Product Consolidation**

* A product's identity is defined by its **technical characteristics** (e.g., "Gypsum Board, Type X, 1/2 inch"), **not by its manufacturer**.
* If you encounter the same product mentioned multiple times with different approved manufacturers, you **MUST** consolidate this information into a **single product entry**.
* **Action:** Aggregate all manufacturers for a single product into its `manufacturers` field. **DO NOT** create a new, separate product object for each manufacturer listed.

---

### **Field Instructions**

1.  **`product_name`** `(string)`
    * **Action:** Create a unique, descriptive name by combining the product's common name with its key identifiers (e.g., ASTM standard, type, grade, size). **Crucially, do not include manufacturer names in the `product_name` itself.**
    * **Example:** `"Carbon Steel Bolts and Studs (ASTM A307 Grade A)"`
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type='application/pdf',
                ),
                prompt],
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant that extracts product information from a construction specification PDF.",
                    max_output_tokens=3000,
                    temperature=0.1,
                    response_mime_type="application/json"
            )
        )
        
        # Parse the JSON response to validate it
        try:
            parsed_json = json.loads(response.text)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON response from API: {e}")
            sys.exit(1)
        
        # Write to output file
        output_path = pathlib.Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully processed '{args.pdf_path}'")
        print(f"Results saved to '{args.output}'")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()