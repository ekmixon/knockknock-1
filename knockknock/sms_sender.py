import os
import datetime
import traceback
import functools
import socket
from twilio.rest import Client


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def sms_sender(account_sid: str, auth_token: str, recipient_number: str, sender_number: str):
    client = Client(account_sid, auth_token)
    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += f" - RANK: {os.environ['RANK']}"
            else:
                master_process = True

            if master_process:
                contents = [
                    'Your training has started 🎬',
                    f'Machine name: {host_name}',
                    f'Main call: {func_name}',
                    f'Starting date: {start_time.strftime(DATE_FORMAT)}',
                ]

                text = '\n'.join(contents)
                client.messages.create(body=text, from_=sender_number, to=recipient_number)

            try:
                value = func(*args, **kwargs)

                if master_process:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = [
                        "Your training is complete 🎉",
                        f'Machine name: {host_name}',
                        f'Main call: {func_name}',
                        f'Starting date: {start_time.strftime(DATE_FORMAT)}',
                        f'End date: {end_time.strftime(DATE_FORMAT)}',
                        f'Training duration: {str(elapsed_time)}',
                    ]


                    try:
                        str_value = str(value)
                        contents.append(f'Main call returned value: {str_value}')
                    except:
                        contents.append('Main call returned value: %s'% "ERROR - Couldn't str the returned value.")

                    text = '\n'.join(contents)
                    client.messages.create(body=text, from_=sender_number, to=recipient_number)

                return value

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = [
                    "Your training has crashed ☠️",
                    f'Machine name: {host_name}',
                    f'Main call: {func_name}',
                    f'Starting date: {start_time.strftime(DATE_FORMAT)}',
                    f'Crash date: {end_time.strftime(DATE_FORMAT)}',
                    'Crashed training duration: %s\n\n' % str(elapsed_time),
                    "Here's the error:",
                    '%s\n\n' % ex,
                    "Traceback:",
                    f'{traceback.format_exc()}',
                ]

                text = '\n'.join(contents)
                client.messages.create(body=text, from_=sender_number, to=recipient_number)
                raise ex

        return wrapper_sender

    return decorator_sender
