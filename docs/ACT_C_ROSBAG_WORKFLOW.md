# ACT C-only rosbag workflow

1. 在 Web 端执行 Reset，确认 Nut C 回到已验证初始场景。
2. 在控制器工作区执行 `git pull`。
3. 采集 raw episode：

   ```bash
   python3 -u tools/collect_act_c_rosbag.py
   ```

4. 输出位于 `data/act_rosbag_raw/episode_YYYYMMDD_HHMMSS_xxxxxx/`，包含 `bag/`、`action_events.jsonl`、`episode_meta.json` 和 `expert.log`。
5. 离线转换：

   ```bash
   python3 -u tools/convert_act_c_rosbag_episode.py --episode data/act_rosbag_raw/<episode_id>
   ```

6. 查看 `<episode>/act_5hz/quality_report.json`；最终结论字段为 `accepted_for_training`。
