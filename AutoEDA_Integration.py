import sweetviz
import webbrowser
from dataprep.eda import create_report

def generate_eda_report(dataset, eda_choice, report_path):
    report_name = eda_choice.lower() + "_report"

    if eda_choice == "Sweetviz":
        embedded_html = """
                    <div class='top-bar' style="background-color:#000000;top:0;left:0;width:126%;border-left:1px;">
                    <img src="https://www2.deloitte.com/content/dam/assets/logos/deloitte.svg" class="topbarlogo" style="width:150px;display:inline-block;padding:20px;margin-left:1.5%;"/></div>
                    """
        sweetviz_path = report_path + "/" + report_name + ".html"
        report = sweetviz.analyze(dataset)
        report.show_html(filepath=sweetviz_path,
                        open_browser=False,  # False
                        layout="widescreen",  # vertical, widescreen
                        scale=None)

        with open(sweetviz_path, 'r+') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('<html lang="en">'):
                    lines[i] = lines[i] + embedded_html
            f.truncate()
            f.seek(0)
            for line in lines:
                f.write(line)
        webbrowser.open_new_tab(sweetviz_path)


    elif eda_choice == "Dataprep":
        embedded_html = """
                    <div class='top-bar' style="background-color:#000000;top:0;left:0;width:100%;border-left:1px;">
                    <img src="https://www2.deloitte.com/content/dam/assets/logos/deloitte.svg" class="topbarlogo" style="width:150px;display:inline-block;padding:20px;margin-left:1.5%;"/></div>
                    """
        dataprep_path = report_path + "/" + report_name + ".html"
        report = create_report(dataset)
        report.save(filename=report_name, to=report_path)
        with open(dataprep_path, encoding="utf8", mode="r+") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('<html lang="en">'):
                    lines[i] = lines[i] + embedded_html
            f.truncate()
            f.seek(0)
            for line in lines:
                f.write(line)
        webbrowser.open_new_tab(dataprep_path)
        #report.show_browser() # openin report in browser