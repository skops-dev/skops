# from __future__ import annotations

# import io
# import tempfile
# from typing import Any, Sequence
# from uuid import uuid4

# import sklearn.exceptions

# from ._audit import Node, get_tree
# from ._utils import LoadContext, SaveContext, get_module, get_state

# try:
#     import xgboost
# except ImportError:
#     xgboost = None  # type: ignore


# def xgboost_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
#     res = {
#         "__class__": obj.__class__.__name__,
#         "__module__": get_module(type(obj)),
#         "__loader__": "XGBoostNode",
#         "constructor": type(obj).__name__,
#     }

#     # fitted vs not fitted models require different approaches
#     try:
#         # xgboost doesn't allow to save to a memory buffer, so take roundtrip
#         # through temp file
#         tmp_file = f"{tempfile.mkdtemp()}.ubj"
#         obj.save_model(tmp_file)
#         with open(tmp_file, "rb") as f:
#             data_buffer = io.BytesIO(f.read())
#         f_name = f"xgboost_{uuid4()}.json"
#         save_context.zip_file.writestr(f_name, data_buffer.getbuffer())
#         res.update(type="xgboost-format", file=f_name)
#     except sklearn.exceptions.NotFittedError:
#         res.update(type="params", content=get_state(obj.get_params(), save_context))

#     return res


# class XGBoostNode(Node):
#     def __init__(
#         self,
#         state: dict[str, Any],
#         load_context: LoadContext,
#         trusted: bool | Sequence[str] = False,
#     ) -> None:
#         super().__init__(state, load_context, trusted)
#         self.type = state["type"]
#         self.trusted = self._get_trusted(trusted, [])

#         # list of constructors is hard-coded for higher security
#         constructors = {
#             "XGBClassifier": xgboost.XGBClassifier,
#             "XGBRegressor": xgboost.XGBRegressor,
#             "XGBRFClassifier": xgboost.XGBRFClassifier,
#             "XGBRFRegressor": xgboost.XGBRFRegressor,
#             "XGBRanker": xgboost.XGBRanker,
#         }
#         self.children = {"constructor": constructors[state["constructor"]]}
#         if self.type == "xgboost-format":  # fitted xgboost model
#             self.children["content"] = io.BytesIO(load_context.src.read(state["file"]))
#         else:
#             self.children["content"] = get_tree(state["content"], load_context)

#     def _construct(self):
#         cls = self.children["constructor"]
#         instance = cls()

#         if self.type == "xgboost-format":  # fitted
#             # load_model works with bytearray, so no temp file necessary here
#             instance.load_model(bytearray(self.children["content"].getvalue()))
#         else:
#             params = self.children["content"].construct()
#             instance.set_params(**params)

#         return instance


GET_STATE_DISPATCH_FUNCTIONS = []  # type: ignore
NODE_TYPE_MAPPING = {}  # type: ignore

# if xgboost is not None:
#     GET_STATE_DISPATCH_FUNCTIONS.append((xgboost.sklearn.XGBModel, xgboost_get_state))
#     NODE_TYPE_MAPPING["XGBoostNode"] = XGBoostNode
