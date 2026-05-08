from crewai import Crew, Process
from core.agents import job_analyst, candidate_retriever, candidate_evaluator, report_writer
from core.tasks import build_tasks
from core.settings_config import settings
import json
import re
import argparse


def parse_output(raw_output: str) -> tuple:
    """
    Parse the crew output to extract JSON and markdown sections.
    Returns (json_data, markdown_report) tuple.
    """
    separator = "=== MARKDOWN REPORT ==="
    if separator in raw_output:
        parts = raw_output.split(separator)
        json_part = parts[0].strip()
        markdown_part = parts[1].strip()
        
        try:
            json_data = json.loads(json_part)
            return json_data, markdown_part
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                    return json_data, raw_output
                except json.JSONDecodeError:
                    pass
            return None, raw_output
    else:
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                return json_data, raw_output
            except json.JSONDecodeError:
                pass
        return None, raw_output


def main():
    parser = argparse.ArgumentParser(
        description="AI HR Resume Screener - Screen and rank job candidates using CrewAI + Qdrant RAG"
    )
    parser.add_argument(
        "--profession",
        type=str,
        required=True,
        help="Profession category to filter candidates (e.g., ENGINEERING, FINANCE, ACCOUNTING)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        choices=range(3, 21),
        metavar="3-20",
        help="Number of top candidates to retrieve (default: 10, range: 3-20)"
    )
    parser.add_argument(
        "--job-description",
        type=str,
        help="Job description text (use this OR --job-desc-file)"
    )
    parser.add_argument(
        "--job-desc-file",
        type=str,
        help="Path to a text file containing the job description"
    )
    
    args = parser.parse_args()
    
    if not args.job_description and not args.job_desc_file:
        parser.error("Either --job-description or --job-desc-file must be provided")
    
    if args.job_desc_file:
        try:
            with open(args.job_desc_file, "r", encoding="utf-8") as f:
                job_description = f.read()
        except FileNotFoundError:
            parser.error(f"Job description file not found: {args.job_desc_file}")
        except Exception as e:
            parser.error(f"Error reading job description file: {e}")
    else:
        job_description = args.job_description
    
    settings.top_k = args.top_k
    
    print("=" * 60)
    print("  AI HR Resume Screener")
    print("  Powered by CrewAI + Qdrant RAG")
    print("=" * 60)
    print(f"Profession: {args.profession}")
    print(f"Top K: {args.top_k}")
    print("=" * 60)
    
    tasks = build_tasks(
        job_description=job_description,
        profession=args.profession,
    )

    crew = Crew(
        agents=[job_analyst, candidate_retriever, candidate_evaluator, report_writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("  FINAL HIRING REPORT")
    print("=" * 60)
    
    json_data, markdown_report = parse_output(str(result.raw))
    
    if json_data:
        with open("hiring_report.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print("\nJSON report saved to hiring_report.json")
    
    with open("hiring_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print("Markdown report saved to hiring_report.md")
    
    print(markdown_report)


if __name__ == "__main__":
    main()
