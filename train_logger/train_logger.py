# Setup TensorBoardX summary writer.
import os
import pickle
from datetime import datetime
import threading

HAS_TB = True
HAS_GS_ENNV = True
HAS_NOTI = True

# It's not a major libraries that are in anaconda, so I will check them for you.
try:
    from tensorboardX import SummaryWriter
except Exception as e:
    #import traceback
    #traceback.print_exc()
    #print("exec. without tensor board.")
    HAS_TB = False

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception as e:
    #import traceback
    #traceback.print_exc()
    #print("exec. without google spreadsheet.")
    HAS_GS_ENNV = False

# just in case
try:
    from notificator import Notificator
except Exception as e:
    #import traceback
    #traceback.print_exc()
    #print("exec. without notificator.")
    HAS_NOTI=False

class TrainLogger(object):
    has_tensorboard = HAS_TB
    has_gs = HAS_GS_ENNV
    has_noti = HAS_NOTI

    # api endpoint
    gs_scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive.file']
    
    # google spreadsheet api limitation?
    # actually I think the default worksheet row, col are size of 1000, 26.
    # so this is migth be the limitation.
    row_limit = 1000
    col_limit = 26

    def __init__(self, file_name, log_dir="./log", default_log_key="log", csv=False, csv_separator=",", use_tensorboradx=False, gspread_credential_path=None, gspread_share_account="", init_row=1, init_col=1, header=False, time_stamp=True, flush_always=True, buff_size=1, notificate=False, blocking=False, suppress_err=True):
        """
            file_name (str) : the file name which you want to use.
                              should be without extension, otherwise it would be like "hoge.csv.csv"
            log_dir (str)   : the saving directory of the log.
                              it would be saved like log_dir/file_name

            csv (bool)          : save the log in .csv file or not.
            csv_separator (str) : separating charactor of csv file.

            use_tensorboradx (bool) : use tensorboadX or not.

            gspread_cred_path (str)         : the paht of google API's credentials json file.
                                              if you set the json file path it enable the logging to the google spreadsheet.
            gspread_share_account (str)     : the sharing account string.
                                              you need set your google account to access to the spreadsheet.
                                              because the created spreadsheet allows the access from API account for initial.
            init_row (int)                  : starting row
            init_col (int)                  : starting col

            header (bool)       : write header in .csv and spreadsheet or not
            time_stamp (bool)   : use time stamp for file name. it's for avoiding overwriting the existing file.
                                  I know you will think like "Hey please check, Is the file name exist or not, and give a new name."
                                  yeah, I'll do it someday
            flush_always (bool) : always flush the log to the file or not.
            buff_size (int)     : buffer size. I'm not sure this is useful...
                                  recommend in size 1, but sometimes you want buff it?.

            notificate (bool)   : use notificator module. you need a setup independently for this.

            blocking (bool) : block until the logging finishes.
                              sometimes google spreadsheed or other things take time,
                              so if you don't want to get disturb, set this False.
                              using thread, for a short interval to write a log, it cause some trouble of order,
                              I recommend to set a order number to log output.

            suppress_err (bool) : suppress the error logs.

        """

        self.log_dir = log_dir
        self.file_name = file_name
        self.default_log_key = default_log_key

        self.pickle_obj = True
        self.set_default_key = False
        self.set_default_tbkey = False

        self.buffer = []
        self.buff_size = buff_size
        self.flush_always = flush_always
        self.time_stamp = time_stamp
        self.blocking = blocking
        self.suppress_err = suppress_err
        
        self.log_dict = {self.default_log_key:[]}

        self.use_csv = csv
        self.csv_separator = csv_separator

        self.header = header

        self.use_tb = use_tensorboradx if self.has_tensorboard else False

        # google spread sheet setting
        self.init_row = init_row if init_row > 0 else 1
        self.init_col = init_col if init_col > 0 else 1
        self.count_row = 0
        self.count_col = 0
        self.sheet_count = 1

        self.csv_logger = None
        self.tb_logger = None
        self.gs_logger = None
        self.gs = None
        self.notificator = None

        self.csv_filename = None
        self.tb_filename = None
        self.gs_filename = None

        if use_tensorboradx and self.has_tensorboard is False:
            if suppress_err == False:
                print("exec. without tensorboeardX.")

        if self.has_noti and notificate:
            self.notificator = Notificator()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if time_stamp:
            self.ts_str = "{}".format(datetime.now().strftime("%Y%m%d_%H-%M-%S"))

            if self.use_csv:
                self.csv_filename = self.file_name+"{}".format("_{}.csv".format(self.ts_str) if time_stamp else ".csv")
                self.csv_logger = open(os.path.join(self.log_dir, self.csv_filename), "w")

            if self.use_tb:
                self.tb_filename = self.file_name+"{}".format("_{}_tb".format(self.ts_str) if time_stamp else "_tb")
                self.tb_logger = SummaryWriter(log_dir=os.path.join(self.log_dir, self.tb_filename))

            if gspread_credential_path is not None and self.has_gs:
                try:
                    credentials = ServiceAccountCredentials.from_json_keyfile_name(gspread_credential_path, self.gs_scope)
                    self.gs = gspread.authorize(credentials)
                    self.gs_filename = self.file_name+"{}".format("_{}".format(datetime.now().strftime("%Y%m%d_%H-%M-%S") if time_stamp else ""))
                    self.gs_sh = self.gs.create(self.gs_filename)

                    # without shareing, we can't access to the spread sheet.
                    self.gs_sh.share(gspread_share_account, perm_type='user', role='writer')

                    # somehow, I can't rename the title of the sheet, so I append new one and delete the default sheet
                    self.gs_logger = self.gs_sh.add_worksheet(title=self.file_name, rows="{}".format(self.row_limit), cols="{}".format(self.col_limit))
                    wks = self.gs_sh.get_worksheet(0)
                    self.gs_sh.del_worksheet(wks)

                except Exception as e:
                    if suppress_err == False:
                        import traceback
                        traceback.print_exc()
                        print("exec. without google spreadsheet.")
                    HAS_GS_ENNV = False

    def __del__(self):
        self.close()

    def log(self, data):
        self.buffer.append(data)

        if len(self.buffer) >= self.buff_size:
            if self.blocking:
                for i in range(len(self.buffer)):
                    self._log(self.buffer.pop(0))
            else:
                try:
                    th = threading.Thread(target=self._thread_log, args=(self.buffer,))
                    th.start()
                    self.buffer = []
                except Exception as e:
                    # for it's a critical part, I won't suppress here
                    import traceback
                    traceback.print_exc()
                    print(e)

    def notify(self, msg, use_thread=True):
        if self.notificator is not None:
            self.notificator.notify(msg, use_thread)

    def set_notificator(self, params=["mail", "slack", "twitter"]):
        if self.notificator is not None:
            for p in params:
                if p.lower() == "mail":
                    self.notificator.setMail()

                elif p.lower() == "slack":
                    self.notificator.setSlack()

                elif p.lower() == "twitter":
                    self.notificator.setTwitter()

    def disable_pickle_object(self):
        # if you not want to save dictonary object of log.
        self.pickle_obj = False

    def set_default_Keys(self, keys):
        """
            because of default logging is dictonary, we need a key.
            this might be not good way for default.
            I might fix it someday.
        """
        self.set_default_key = True
        self.default_dic_key = keys

        if self.use_csv:
            self._csv_header(keys)
            if self.flush_always:
                self.csv_logger.flush()
        if self.gs is not None:
            self._gs_header(keys)

    def set_default_tbkeys(self, keys):
        """
            tensorboeard has different namespace system, 
            I separete it.
        """
        if self.use_tb:
            self.set_default_tbkey = True
            self.default_dic_tbkey = keys

    def show_keys(self):
        print(self.default_dic_key)

    def keys(self):
        return self.default_dic_key

    def addAppendLogDic(self, key, value):
        """
            thiking for adding information like
            how many epochs or starting time and so on.
            not for the sequential data.
        """
        self.log_dict[key] = value

    def close(self):
        """
            in some case, it might cause a messing log order beecause of using threads
            I am not waiting other threads to be finished now.
            for avoiding this case, I recommend using buffer size to be 1.
        """
        for i in range(len(self.buffer)):
            self._log(self.buffer.pop(0))

        if self.pickle_obj: 
            with open(os.path.join(self.log_dir, self.file_name+"{}".format("_{}.pkl".format(datetime.now().strftime("%Y%m%d_%H-%M-%S") if self.time_stamp else ".pkl"))), "wb") as f:
                pickle.dump(self.log_dict, f)

        if self.csv_logger is not None:
            self.csv_logger.close()

        if self.tb_logger is not None:
            self.tb_logger.close()

    def _csv_header(self, keys):
        for key in keys:
            if key != keys[0]:
                self.csv_logger.write(self.csv_separator)
            self.csv_logger.write(str(key))
        self.csv_logger.write("\n")
        if self.flush_always:
            self.csv_logger.flush()

    def _gs_header(self, keys):
        self.count_col = 0
        for key in keys:
            self.gs_logger.update_cell(self.init_row+self.count_row, self.init_col+self.count_col, str(key))
            self.count_col += 1

        self.count_row += 1

    def _log(self, data):
        if self.pickle_obj:
            self._log_dic(data)

        if self.use_csv:
            self._log_csv(data)
        
        if self.use_tb:
            self._log_tb(data)

        if self.gs is not None:
            self._log_gs(data)

    def _thread_log(self, data):
        for d in data:
            self._log(d)

    def _log_dic(self, data):
        assert len(self.default_dic_key) == len(data), "keys and data has not same length."
        assert self.set_default_key, "no default keys"

        self.log_dict[self.default_log_key].append({})

        for key, val in zip(self.default_dic_key, data):
            self.log_dict[self.default_log_key][-1][key] = val

    def _log_csv(self, data):
        for key, val in zip(self.default_dic_key, data):
            if key != self.default_dic_key[0]:
                self.csv_logger.write(self.csv_separator)
            self.csv_logger.write(str(val))
        self.csv_logger.write("\n")
        if self.flush_always:
            self.csv_logger.flush()

    def _log_tb(self, data):
        assert len(self.default_dic_tbkey) == len(data), "keys and data has not same length."
        assert self.set_default_tbkey, "no default keys"

        for key, val in zip(self.default_dic_key, data):
            self.tb_logger.add_scalar(key, val)

    def _log_gs(self, data):
        """
            cell is
                A B C
               ------>
            1 |
            2 |
            3 v

            which is ('B1') -> (1, 2)
        """
        self.count_col = 0
        for d in data:
            self.gs_logger.update_cell(self.init_row+self.count_row, self.init_col+self.count_col, d)
            self.count_col += 1
            if self.count_col >= self.row_limit:
                self.sheet_count += 1
                self.gs_logger = self.gs_sh.add_worksheet(title=self.file_name+"_{}".format(self.sheet_count), rows="{}".format(self.row_limit), cols="{}".format(self.col_limit))

        self.count_row += 1

    # not thinking to use
    def _write_log(self, key, value):
        self.log_dict[key] = value

    # not thinking to use
    def _write_csvlog(self, data):
        if self.use_csv:
            self.csv_logger.write(data)
            if self.flush_always:
                self.csv_logger.flush()

    def _write_tblog(self, name_space, val, iter_num=None):
        if self.use_tb:
            if iter_num is not None:
                self.tb_logger.add_scalar(name_space, val, iter_num)
            else:
                self.tb_logger.add_scalar(name_space, val)
