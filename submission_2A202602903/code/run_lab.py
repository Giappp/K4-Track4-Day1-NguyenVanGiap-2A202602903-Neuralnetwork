"""Run the notebook sequentially in IPython and preserve real cell outputs.

No Jupyter server required. Any cell exception fails the command.
"""
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path=Path(__file__).resolve().parent/'lab.ipynb'
    nb=nbformat.read(path,as_version=4)
    shell=InteractiveShell.instance()
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':
            continue
        count+=1
        print(f'Executing notebook cell {count}...',flush=True)
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=True)
        outputs=[]
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        cell.outputs=outputs
        cell.execution_count=count
        nbformat.write(nb,path)
        print(captured.stdout[-1200:],flush=True)
        if not result.success:
            raise RuntimeError(f'Notebook cell {count} failed') from (result.error_in_exec or result.error_before_exec)
    print('Notebook executed successfully:',path,flush=True)

if __name__=='__main__':
    main()
