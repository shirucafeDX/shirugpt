"""Create an isolated dataset for native usage logs; grant the service agent write access."""
import argparse
import json
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from src.config import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-number', required=True)
    parser.add_argument('--dataset', default='shirugpt_analytics_logs')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not args.project_number.isdigit():
        parser.error('project-number must be numeric')
    settings = load_settings()
    if args.dataset in settings.allowed_datasets or args.dataset == settings.log_dataset:
        parser.error('分析データ・既存管理ログとは別の専用データセットを指定してください。')
    client = bigquery.Client(project=settings.gcp_project)
    source = client.get_dataset(f'{settings.gcp_project}.{settings.default_dataset}')
    name = f'{settings.gcp_project}.{args.dataset}'
    principal = f'service-{args.project_number}@gcp-sa-geminidataanalytics.iam.gserviceaccount.com'
    print(json.dumps({'dataset': name, 'location': source.location,
                      'writer': principal, 'apply': args.apply}), flush=True)
    if not args.apply:
        return
    try:
        dataset = client.get_dataset(name)
    except NotFound:
        dataset = bigquery.Dataset(name)
        dataset.location = source.location
        dataset.description = 'ShiruGPT native conversational analytics usage logs. Not an analysis data source.'
        dataset.default_table_expiration_ms = 30 * 24 * 3600 * 1000
        dataset = client.create_dataset(dataset)
    entry = bigquery.AccessEntry('WRITER', 'userByEmail', principal)
    if entry not in dataset.access_entries:
        dataset.access_entries = [*dataset.access_entries, entry]
        client.update_dataset(dataset, ['access_entries'])
    print('Native trace dataset and service-agent access configured.')


if __name__ == '__main__':
    main()
