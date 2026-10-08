"""One command, isolated run directory, reproducible evidence and readable failures."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import uuid
import numpy as np
import pandas as pd
from src.schema import ValidationError, validate
from src.load_registry import InputFileError, find_input_file, load_registry, write_outputs
from src.classify_errors import format_report as format_p5_report, run_p5
from src.report import build_report
from src.matching import run_matching
from src.balance import run_balance
from src.ml_risk import run_ml, MLUnavailableError

PROJECT=Path(__file__).resolve().parent


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for stream in self.streams: stream.write(text); stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def check_inputs(data_dir):
    folder=Path(data_dir).resolve()
    files={kind:find_input_file(folder,kind+'.csv' if kind!='registry' else 'registry*.csv')
           for kind in ['acts','etm','registry']}
    for kind,path in files.items():
        try: header=pd.read_csv(path,nrows=1,encoding='utf-8-sig')
        except (UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError) as exc:
            raise InputFileError(f'Не удалось прочитать {path.name}: {exc}') from exc
        validate(header,kind)
    return files


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description='Сверка 1С/ETM/реестра и отчет для бухгалтера')
    parser.add_argument('--data',required=True,help='Папка: acts.csv, etm.csv, один registry*.csv')
    parser.add_argument('--out',required=True,help='Папка результата; каждый запуск сохраняется отдельно')
    parser.add_argument('--verified-labels',help='CSV независимой разметки ML')
    parser.add_argument('--skip-ml',action='store_true',help='Сформировать сверку без обучения модели')
    return parser.parse_args(argv)


def environment():
    versions={}
    for package in ['pandas','numpy','openpyxl','pyarrow','pytest']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='not installed'
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':versions}


def run_pipeline(data_dir,out_dir,verified_labels=None,skip_ml=False):
    data=Path(data_dir).resolve(); out=Path(out_dir).resolve()
    if out.is_relative_to(data): raise InputFileError('Папка результата должна находиться вне папки исходных данных')
    files=check_inputs(data)
    if verified_labels and not Path(verified_labels).is_file(): raise InputFileError('Не найден CSV независимой разметки ML')
    out.mkdir(parents=True,exist_ok=True)
    identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    run=out/'runs'/identifier; interim=run/'interim'; interim.mkdir(parents=True)
    info={'run_id':identifier,'status':'running','seed':42,'environment':environment(),
          'inputs':{kind:{'path':str(path),'sha256':sha256(path)} for kind,path in files.items()},
          'source_sha256':{str(path.relative_to(PROJECT)):sha256(path) for path in
                           sorted(list((PROJECT/'src').glob('*.py'))+[PROJECT/'reconcile.py',PROJECT/'cleaning.py'])},
          'parameters':{'skip_ml':skip_ml,'verified_labels':str(Path(verified_labels).resolve()) if verified_labels else None}}
    if verified_labels: info['verified_labels_sha256']=sha256(verified_labels)
    def manifest(): (run/'run_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest()
    with (run/'run.log').open('w',encoding='utf-8') as log, redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
        try:
            random.seed(42); np.random.seed(42)
            print(f'Запуск: {identifier}\nРезультаты и диагностика: {run}')
            print('\nЭтап 1: проверка и очистка выгрузок')
            clean=interim/'clean'; matching=interim/'reconciliation'
            env=os.environ.copy(); env['PYTHONUTF8']='1'
            process=subprocess.run([sys.executable,str(PROJECT/'cleaning.py'),str(data),str(clean)],
                                   capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
            print(process.stdout)
            if process.returncode:
                print(process.stderr)
                raise ValidationError(f'Очистка остановлена. Проверьте {clean / "data_quality_issues.csv"} и журнал запуска')
            registry=load_registry(data); write_outputs(registry,interim)
            print('\nЭтап 2: сопоставление операций')
            run_matching(clean,interim/'registry.parquet',matching)
            print('\nЭтап 3: мост баланса')
            run_balance(clean,matching)
            print('\nЭтапы 4–5: классификация и аномалии')
            result=run_p5(clean,interim/'registry.parquet',interim/'p5',require_reconciliation=True)
            print(format_p5_report(result))
            print('\nЭтап 7: модель риска')
            ml_status={'status':'skipped','reason':'Обучение отключено параметром --skip-ml'}
            if not skip_ml:
                try:
                    metrics=run_ml(interim,verified_labels)
                    ml_status={'status':'trained','label_basis':metrics['label_basis']}
                except MLUnavailableError as exc:
                    ml_status={'status':'unavailable','reason':str(exc)}
                    print(f'Модель не обучена: {exc}. Сверка и отчет будут сформированы.')
            (interim/'ml').mkdir(exist_ok=True)
            (interim/'ml/status.json').write_text(json.dumps(ml_status,ensure_ascii=False,indent=2),encoding='utf-8')
            info['ml']=ml_status
            print('\nЭтап 6: действия бухгалтера и итоговый Excel')
            report=build_report(interim,run/'reconciliation_report.xlsx')
            # Atomic publication: a failed run preserves the last successful report.
            temporary=out/(identifier+'.xlsx.tmp')
            shutil.copyfile(report,temporary); os.replace(temporary,out/'reconciliation_report.xlsx')
            info['status']='completed'; info['report']=str(report)
            info['outputs_sha256']={str(path.relative_to(run)):sha256(path) for path in sorted(interim.rglob('*'))
                                    if path.is_file() and path.suffix in ['.csv','.json']}
            manifest()
            pointer=out/(identifier+'.json.tmp')
            pointer.write_text(json.dumps({'run_id':identifier,'run_dir':str(run),'report':str(report),'ml':ml_status},ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(pointer,out/'latest_run.json')
            print(f'\nГотово. Откройте {out / "reconciliation_report.xlsx"}')
            print(f'Архив запуска: {run}')
        except Exception as exc:
            info['status']='failed'; info['error']=str(exc); manifest()
            print(f'Запуск остановлен: {exc}\nДиагностика сохранена: {run / "run.log"}')
            raise
    return run


def main(argv=None):
    args=parse_args(argv)
    try:
        run_pipeline(args.data,args.out,args.verified_labels,args.skip_ml)
        return 0
    except (ValidationError,InputFileError,PermissionError,FileNotFoundError,ImportError) as exc:
        print(f'Ошибка: {exc}',file=sys.stderr)
        if isinstance(exc,PermissionError): print('Закройте Excel, проверьте доступ к папке результата и повторите запуск.',file=sys.stderr)
        if isinstance(exc,ImportError): print('Установите зависимости: python -m pip install -r requirements-lock.txt',file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Неожиданная ошибка: {exc}. Подробности находятся в run.log последнего запуска.',file=sys.stderr)
        return 1


if __name__=='__main__': sys.exit(main())
