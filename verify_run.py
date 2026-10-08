"""Check run integrity and compare deterministic data outputs, not ZIP timestamps."""
import argparse
import json
from pathlib import Path
from reconcile import sha256


def verify(run_dir):
    root=Path(run_dir)
    manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed': raise ValueError('Запуск не завершен успешно')
    broken=[name for name,value in manifest['outputs_sha256'].items()
            if not (root/name).is_file() or sha256(root/name)!=value]
    if broken: raise ValueError('Измененные/отсутствующие результаты: '+', '.join(broken))
    return manifest


def compare(first,second):
    a,b=verify(first),verify(second)
    if {k:v['sha256'] for k,v in a['inputs'].items()}!={k:v['sha256'] for k,v in b['inputs'].items()}:
        raise ValueError('Исходные данные различаются')
    if a['source_sha256']!=b['source_sha256']: raise ValueError('Версии кода различаются')
    if a['parameters']!=b['parameters']: raise ValueError('Параметры различаются')
    if a['environment']!=b['environment']: raise ValueError('Окружения различаются')
    names=set(a['outputs_sha256'])|set(b['outputs_sha256'])
    changed=[name for name in sorted(names) if a['outputs_sha256'].get(name)!=b['outputs_sha256'].get(name)]
    if changed: raise ValueError('Результаты различаются: '+', '.join(changed))
    return len(names)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Проверка целостности и повторяемости запуска')
    parser.add_argument('run_dir'); parser.add_argument('--against')
    args=parser.parse_args(argv)
    try:
        if args.against: print(f'Совпадают {compare(args.run_dir,args.against)} файлов данных и модели')
        else: print(f'Целостность подтверждена: {len(verify(args.run_dir)["outputs_sha256"])} файлов')
        return 0
    except (ValueError,OSError,KeyError) as exc:
        print(f'Проверка не пройдена: {exc}'); return 1


if __name__=='__main__': raise SystemExit(main())
