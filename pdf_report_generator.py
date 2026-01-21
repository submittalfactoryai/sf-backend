"""
PDF Report Generator Module
Handles generation of validation reports as PDF pages and merging with existing PDFs
"""

import io
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List
from pypdf import PdfWriter, PdfReader
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, green, red, orange, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
import requests
import logging

logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """Generates validation reports as PDF pages with enhanced UI"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Main title style with blue color
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=28,
            textColor=HexColor('#1e40af'),
            spaceAfter=30,
            spaceBefore=10,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=HexColor('#6b7280'),
            spaceAfter=20,
            alignment=1,  # Center
            fontName='Helvetica'
        ))
        
        # Section header style with background-like appearance
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#1f2937'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            leftIndent=10,
            rightIndent=10
        ))
        
        # Score style with large, bold text
        self.styles.add(ParagraphStyle(
            name='ScoreStyle',
            parent=self.styles['Normal'],
            fontSize=36,
            textColor=HexColor('#1e40af'),
            spaceAfter=15,
            spaceBefore=10,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        ))
        
        # Enhanced normal text style
        self.styles.add(ParagraphStyle(
            name='EnhancedNormal',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            leading=16,
            fontName='Helvetica',
            textColor=HexColor('#374151')
        ))
        
        # Success item style (green checkmark items)
        self.styles.add(ParagraphStyle(
            name='SuccessItem',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leading=15,
            fontName='Helvetica',
            textColor=HexColor('#065f46'),
            leftIndent=20,
            bulletIndent=10
        ))
        
        # Warning item style (red X items)
        self.styles.add(ParagraphStyle(
            name='WarningItem',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leading=15,
            fontName='Helvetica',
            textColor=HexColor('#7f1d1d'),
            leftIndent=20,
            bulletIndent=10
        ))
        
        # Info box style
        self.styles.add(ParagraphStyle(
            name='InfoBox',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            spaceBefore=12,
            leading=16,
            fontName='Helvetica',
            textColor=HexColor('#1f2937'),
            leftIndent=15,
            rightIndent=15,
            borderColor=HexColor('#e5e7eb'),
            borderWidth=1,
            borderPadding=10
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=HexColor('#9ca3af'),
            alignment=1,  # Center
            fontName='Helvetica-Oblique'
        ))

    def _create_header_section(self) -> List:
        """Create an enhanced header section with styling"""
        content = []
        
        # Main title with better styling
        title = Paragraph("SUBMITTAL VALIDATION REPORT", self.styles['ReportTitle'])
        content.append(title)
        
        # Subtitle
        subtitle = Paragraph("Automated Specification Verification", self.styles['Subtitle'])
        content.append(subtitle)
        content.append(Spacer(1, 20))
        
        return content

    def _create_info_table(self, validation_data: Dict[str, Any], product_name: str) -> Table:
        """Create a professional info table with validation details"""
        # Prepare table data
        table_data = [
            ['Product Name', product_name],
            ['Validation Date', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Validation Status', validation_data.get('valid', 'Unknown')],
            ['Product Name Found', validation_data.get('product_name_found', 'Unknown')],
            ['Specifications Match', validation_data.get('specifications_match', '0/0')],
            ['Manufacturers Found', validation_data.get('any_manufacturer_found', 'Unknown')]
        ]
        
        # Create table
        table = Table(table_data, colWidths=[2.5*inch, 3.5*inch])
        
        # Apply styling
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#374151')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            
            # Cell styling
            ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#1f2937')),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            
            # Borders and alignment
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [white, HexColor('#f9fafb')]),
            
            # Padding
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        return table

    def _create_score_section(self, score: int) -> List:
        """Create an enhanced score display section"""
        content = []
        
        # Score header
        score_header = Paragraph("VALIDATION SCORE", self.styles['SectionHeader'])
        content.append(score_header)
        
        # Large score display with color coding
        score_color = self._get_score_color(score)
        score_style = ParagraphStyle(
            name='ColoredScore',
            parent=self.styles['ScoreStyle'],
            textColor=HexColor(score_color)
        )
        
        score_para = Paragraph(f"<b>{score}%</b>", score_style)
        content.append(score_para)
        
        # Score interpretation
        interpretation = self._get_score_interpretation(score)
        interp_para = Paragraph(f"<i>{interpretation}</i>", self.styles['EnhancedNormal'])
        content.append(interp_para)
        content.append(Spacer(1, 20))
        
        return content

    def _create_specifications_section(self, validation_data: Dict[str, Any]) -> List:
        """Create an enhanced specifications section"""
        content = []
        
        matched_specs = validation_data.get('matched_specifications', [])
        unmatched_specs = validation_data.get('unmatched_specifications', [])
        
        if matched_specs:
            # Matched specifications with green styling
            header = Paragraph(f"✅ MATCHED SPECIFICATIONS ({len(matched_specs)})", 
                             self.styles['SectionHeader'])
            content.append(header)
            
            for i, spec in enumerate(matched_specs, 1):
                spec_para = Paragraph(f"<b>{i}.</b> {spec}", self.styles['SuccessItem'])
                content.append(spec_para)
            
            content.append(Spacer(1, 15))
        
        if unmatched_specs:
            # Unmatched specifications with red styling
            header = Paragraph(f"❌ UNMATCHED SPECIFICATIONS ({len(unmatched_specs)})", 
                             self.styles['SectionHeader'])
            content.append(header)
            
            for i, spec in enumerate(unmatched_specs, 1):
                spec_para = Paragraph(f"<b>{i}.</b> {spec}", self.styles['WarningItem'])
                content.append(spec_para)
            
            content.append(Spacer(1, 15))
        
        return content

    def _create_manufacturer_section(self, validation_data: Dict[str, Any]) -> List:
        """Create an enhanced manufacturer section"""
        content = []
        
        found_manufacturers = validation_data.get('found_manufacturers', [])
        
        if found_manufacturers:
            header = Paragraph("🏢 MANUFACTURER VALIDATION", self.styles['SectionHeader'])
            content.append(header)
            
            mfg_list = ", ".join(found_manufacturers)
            mfg_para = Paragraph(f"<b>Listed Manufacturers Found:</b> {mfg_list}", 
                               self.styles['SuccessItem'])
            content.append(mfg_para)
        else:
            # Check if validation was attempted (unmatched manufacturers exist)
            unmatched_manufacturers = validation_data.get('unmatched_manufacturers', [])
            if unmatched_manufacturers:
                header = Paragraph("🏢 MANUFACTURER VALIDATION", self.styles['SectionHeader'])
                content.append(header)
                
                info_para = Paragraph("No manufacturers from the approved list were found in this document.", 
                                    self.styles['InfoBox'])
                content.append(info_para)
        
        content.append(Spacer(1, 15))
        return content

    def generate_validation_report_pdf(self, validation_data: Dict[str, Any], product_name: str) -> bytes:
        """
        Generate an enhanced validation report as a PDF page
        
        Args:
            validation_data: The validation results from LLM
            product_name: Name of the product being validated
            
        Returns:
            bytes: PDF content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            topMargin=0.5*inch, 
            bottomMargin=0.5*inch, 
            leftMargin=0.75*inch, 
            rightMargin=0.75*inch
        )
        
        # Build the enhanced content
        content = []
        
        # Header section
        content.extend(self._create_header_section())
        
        # Information table
        info_table = self._create_info_table(validation_data, product_name)
        content.append(info_table)
        content.append(Spacer(1, 25))
        
        # Score section
        score = validation_data.get('validation_score', 0)
        content.extend(self._create_score_section(score))
        
        # Executive summary
        summary = validation_data.get('summary', 'No summary available')
        if summary and summary != 'No summary available':
            summary_header = Paragraph("📋 EXECUTIVE SUMMARY", self.styles['SectionHeader'])
            content.append(summary_header)
            summary_para = Paragraph(summary, self.styles['InfoBox'])
            content.append(summary_para)
            content.append(Spacer(1, 20))
        
        # Specifications section
        content.extend(self._create_specifications_section(validation_data))
        
        # Manufacturer section
        content.extend(self._create_manufacturer_section(validation_data))
        
        # Technical details section
        self._add_technical_details(content, validation_data)
        
        # Enhanced footer
        content.append(Spacer(1, 30))
        footer_line = HRFlowable(width="100%", thickness=2, color=HexColor('#e5e7eb'))
        content.append(footer_line)
        content.append(Spacer(1, 10))
        
        footer_para = Paragraph(
            f"Generated by <b>Submittal Factory</b> Validation System<br/>"
            f"Report ID: {datetime.now().strftime('%Y%m%d-%H%M%S')}", 
            self.styles['Footer']
        )
        content.append(footer_para)
        
        # Build PDF
        doc.build(content)
        buffer.seek(0)
        return buffer.getvalue()

    def _add_technical_details(self, content: List, validation_data: Dict[str, Any]):
        """Add technical details section if available"""
        elapsed_time = validation_data.get('elapsed_time')
        model_used = validation_data.get('model_used')
        
        if elapsed_time or model_used:
            tech_header = Paragraph("🔧 TECHNICAL DETAILS", self.styles['SectionHeader'])
            content.append(tech_header)
            
            if elapsed_time:
                time_para = Paragraph(f"<b>Processing Time:</b> {elapsed_time:.2f} seconds", 
                                    self.styles['EnhancedNormal'])
                content.append(time_para)
            
            if model_used:
                model_para = Paragraph(f"<b>AI Model:</b> {model_used}", 
                                     self.styles['EnhancedNormal'])
                content.append(model_para)
            
            content.append(Spacer(1, 15))
    
    def _get_score_color(self, score: int) -> str:
        """Get color based on validation score"""
        if score >= 80:
            return "#059669"  # Emerald green
        elif score >= 60:
            return "#d97706"  # Amber orange
        else:
            return "#dc2626"  # Red
    
    def _get_score_interpretation(self, score: int) -> str:
        """Get interpretation text based on score"""
        if score >= 90:
            return "Excellent Match - Highly Recommended"
        elif score >= 80:
            return "Good Match - Recommended"
        elif score >= 60:
            return "Partial Match - Review Required"
        elif score >= 40:
            return "Poor Match - Not Recommended"
        else:
            return "No Match - Reject"


class PDFMerger:
    """Handles merging of validation reports with existing PDFs"""
    
    @staticmethod
    def merge_report_with_pdf(report_pdf_bytes: bytes, original_pdf_url: str = None, original_pdf_bytes: bytes = None) -> bytes:
        """
        Merge validation report as first page with original PDF
        
        Args:
            report_pdf_bytes: Validation report PDF as bytes
            original_pdf_url: URL to the original PDF (optional if original_pdf_bytes provided)
            original_pdf_bytes: Original PDF content as bytes (optional if original_pdf_url provided)
            
        Returns:
            bytes: Merged PDF content
        """
        try:
            # Get original PDF bytes - either from provided bytes or download from URL
            if original_pdf_bytes:
                # Use provided bytes directly (avoids second download)
                target_pdf_bytes = original_pdf_bytes
            elif original_pdf_url:
                # Download from URL as fallback
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(original_pdf_url, timeout=300, headers=headers)
                response.raise_for_status()
                target_pdf_bytes = response.content
            else:
                raise ValueError("Either original_pdf_url or original_pdf_bytes must be provided")
            
            # Create PDF readers
            report_reader = PdfReader(io.BytesIO(report_pdf_bytes))
            original_reader = PdfReader(io.BytesIO(target_pdf_bytes))
            
            # Create writer for merged PDF
            writer = PdfWriter()
            
            # Add report pages first
            for page in report_reader.pages:
                writer.add_page(page)
            
            # Add original PDF pages
            for page in original_reader.pages:
                writer.add_page(page)
            
            # Write to bytes
            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            output_buffer.seek(0)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            print(f"Error merging PDFs: {str(e)}")
            logger.warning(f"PDF merge failed: {str(e)}. Attempting fallback strategies.")
            
            # Fallback 1: Return just the validation report if merge fails
            try:
                return report_pdf_bytes
            except:
                pass
            
            # Fallback 2: Try to return original PDF if available
            try:
                if original_pdf_bytes:
                    logger.info("Returning original PDF content as fallback")
                    return original_pdf_bytes
                elif original_pdf_url:
                    logger.info("Attempting to re-download PDF as fallback")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(original_pdf_url, timeout=300, headers=headers)
                    response.raise_for_status()
                    return response.content
                else:
                    logger.warning("No fallback PDF content available")
                    return report_pdf_bytes  # Return just the validation report
            except Exception as fallback_error:
                logger.error(f"All fallback strategies failed: {fallback_error}")
                # Last resort: return just the validation report
                return report_pdf_bytes


def generate_validation_report_and_merge(
    validation_data: Dict[str, Any], 
    product_name: str, 
    original_pdf_url: str = None,
    original_pdf_bytes: bytes = None
) -> bytes:
    """
    Complete workflow: generate validation report and merge with original PDF
    
    Args:
        validation_data: Validation results from LLM
        product_name: Name of the product
        original_pdf_url: URL to original PDF (optional if original_pdf_bytes provided)
        original_pdf_bytes: Original PDF content as bytes (optional if original_pdf_url provided)
        
    Returns:
        bytes: Merged PDF with validation report as first page
    """
    generator = ValidationReportGenerator()
    merger = PDFMerger()
    
    # Generate validation report PDF
    report_pdf = generator.generate_validation_report_pdf(validation_data, product_name)
    
    # Merge with original PDF
    merged_pdf = merger.merge_report_with_pdf(
        report_pdf, 
        original_pdf_url=original_pdf_url,
        original_pdf_bytes=original_pdf_bytes
    )
    
    return merged_pdf 